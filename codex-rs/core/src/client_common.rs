pub use codex_api::ResponseEvent;
use codex_protocol::ResponseItemId;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::DEFAULT_IMAGE_DETAIL;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ImageDetail;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::plaintext_agent_message_content;
use codex_protocol::openai_models::ModelInfo;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ToolSpec;
use futures::Stream;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Core inputs to provider request formatting for a single model turn.
/// The rollout trace serializes this shape before transport-specific rewrites.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Prompt {
    /// Conversation context input items.
    #[serde(serialize_with = "codex_rollout_trace::serialize_trace_items")]
    pub input: Vec<ResponseItem>,

    /// Tool definitions to inject into this request, empty when supplied by history.
    pub(crate) tools: Arc<[ToolSpec]>,

    /// Whether parallel tool calls are permitted for this prompt.
    pub(crate) parallel_tool_calls: bool,

    pub base_instructions: BaseInstructions,

    /// Optional the output schema for the model's response.
    pub output_schema: Option<Value>,

    /// Whether the Responses API should strictly validate `output_schema`.
    pub output_schema_strict: bool,

    pub(crate) prompt_cache_ttl: Option<crate::PromptCacheTtl>,

    pub(crate) cyber_access_program: Option<codex_protocol::turn_input::CyberAccessProgram>,
}

impl Default for Prompt {
    fn default() -> Self {
        Self {
            input: Vec::new(),
            tools: Arc::default(),
            parallel_tool_calls: false,
            base_instructions: BaseInstructions::default(),
            output_schema: None,
            output_schema_strict: true,
            prompt_cache_ttl: None,
            cyber_access_program: None,
        }
    }
}

impl Prompt {
    pub(crate) fn get_formatted_input_for_request(
        &self,
        model_info: &ModelInfo,
    ) -> Vec<ResponseItem> {
        let mut input = self.input.clone();
        normalize_image_details(&mut input, model_info);
        input
    }
}

/// Remove OpenAI-private control fields before a request reaches an external provider.
pub(crate) fn normalize_input_for_non_openai_provider(
    input: Vec<ResponseItem>,
) -> Result<Vec<ResponseItem>> {
    let mut normalized = Vec::with_capacity(input.len());
    for item in input {
        let Some(mut item) = normalize_item_for_non_openai_provider(item)? else {
            continue;
        };
        item.clear_internal_chat_message_metadata_passthrough();
        if let ResponseItem::FunctionCall {
            encrypted_function_args,
            ..
        } = &mut item
        {
            *encrypted_function_args = None;
        }
        normalized.push(item);
    }
    Ok(normalized)
}

fn normalize_item_for_non_openai_provider(item: ResponseItem) -> Result<Option<ResponseItem>> {
    let (id, content) = match item {
        ResponseItem::AgentMessage { id, content, .. } => (id, content),
        ResponseItem::ConfigurationUpdate { .. } => return Ok(None),
        item => return Ok(Some(item)),
    };
    let text = plaintext_agent_message_content(&content).ok_or_else(|| {
        CodexErr::InvalidRequest(
            "non-OpenAI model providers cannot consume encrypted inter-agent messages".to_string(),
        )
    })?;
    let id = id.map(|id| {
        let suffix = id.as_str().strip_prefix("amsg_").unwrap_or(id.as_str());
        ResponseItemId::with_suffix("msg", suffix)
    });
    Ok(Some(ResponseItem::Message {
        id,
        role: "user".to_string(),
        content: vec![ContentItem::InputText { text }],
        phase: None,
        status: None,
        encrypted_content: None,
        internal_chat_message_metadata_passthrough: None,
    }))
}

/// Meta Muse models reject the entire request when *any* advertised tool has
/// more than 10 nested JSON-schema levels or unresolved references. Preserve
/// all other tools, including compatible tools in a partially rejected namespace.
/// Do not change schemas silently: excluded tools are not advertised at all.
pub(crate) fn tool_specs_for_muse_spark(tools: &[ToolSpec]) -> Vec<ToolSpec> {
    let mut excluded = 0usize;
    let mut kept = Vec::with_capacity(tools.len());

    for spec in tools {
        match spec {
            ToolSpec::Function(function) => {
                if muse_schema_supported(&function.parameters) {
                    kept.push(spec.clone());
                } else {
                    excluded += 1;
                }
            }
            ToolSpec::Namespace(namespace) => {
                let mut namespace = namespace.clone();
                namespace.tools.retain(|tool| match tool {
                    ResponsesApiNamespaceTool::Function(function) => {
                        let supported = muse_schema_supported(&function.parameters);
                        if !supported {
                            excluded += 1;
                        }
                        supported
                    }
                    ResponsesApiNamespaceTool::Custom(_) => true,
                });
                if !namespace.tools.is_empty() {
                    kept.push(ToolSpec::Namespace(namespace));
                }
            }
            ToolSpec::ToolSearch { parameters, .. } => {
                if muse_schema_supported(parameters) {
                    kept.push(spec.clone());
                } else {
                    excluded += 1;
                }
            }
            ToolSpec::WebSearch { .. } | ToolSpec::Freeform(_) => kept.push(spec.clone()),
        }
    }

    if excluded > 0 {
        tracing::warn!(
            excluded_tools = excluded,
            "Meta Muse tool schemas exceed the provider nesting/reference limits; omitted only the incompatible definitions"
        );
    }
    kept
}

fn muse_schema_supported<T: serde::Serialize>(schema: &T) -> bool {
    match serde_json::to_value(schema) {
        Ok(value) => muse_schema_within_limits(&value, 1),
        Err(_) => false,
    }
}

fn muse_schema_within_limits(schema: &Value, depth: usize) -> bool {
    // A hard cap is cheaper and safer than a lossy schema rewrite. Keep
    // argument validation intact for every tool still shown to the model.
    if depth > 10 {
        return false;
    }
    match schema {
        Value::Object(map) => {
            !map.contains_key("$ref")
                && map
                    .values()
                    .all(|value| muse_schema_within_limits(value, depth + 1))
        }
        Value::Array(values) => values
            .iter()
            .all(|value| muse_schema_within_limits(value, depth + 1)),
        _ => true,
    }
}

fn normalize_image_details(items: &mut [ResponseItem], model_info: &ModelInfo) {
    for item in items {
        match item {
            ResponseItem::Message { content, .. } => {
                for content_item in content {
                    if let ContentItem::InputImage { detail, .. } = content_item {
                        normalize_image_detail(detail, model_info);
                    }
                }
            }
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => {
                if let Some(content) = output.content_items_mut() {
                    for content_item in content {
                        if let FunctionCallOutputContentItem::InputImage { detail, .. } =
                            content_item
                        {
                            normalize_image_detail(detail, model_info);
                        }
                    }
                }
            }
            ResponseItem::AdditionalTools { .. }
            | ResponseItem::Reasoning { .. }
            | ResponseItem::AgentMessage { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::FunctionCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::ConfigurationUpdate { .. }
            | ResponseItem::CompactionTrigger { .. }
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {}
        }
    }
}

fn normalize_image_detail(detail: &mut Option<ImageDetail>, model_info: &ModelInfo) {
    if model_info.use_responses_lite {
        *detail = None;
    } else if *detail == Some(ImageDetail::Original) && !model_info.supports_image_detail_original {
        *detail = Some(DEFAULT_IMAGE_DETAIL);
    }
}

pub struct ResponseStream {
    pub(crate) routing_affinity: Option<codex_http_client::ChatGptRoutingAffinity>,
    pub(crate) rx_event: mpsc::Receiver<Result<ResponseEvent>>,
    pub(crate) interrupt: Option<oneshot::Sender<codex_api::ResponseInterrupt>>,
    /// Signals the mapper task that the consumer stopped polling before the
    /// provider stream reached its own terminal event.
    pub(crate) consumer_dropped: CancellationToken,
}

impl Stream for ResponseStream {
    type Item = Result<ResponseEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx_event.poll_recv(cx)
    }
}

impl Drop for ResponseStream {
    fn drop(&mut self) {
        self.consumer_dropped.cancel();
    }
}

#[cfg(test)]
#[path = "client_common_tests.rs"]
mod tests;
