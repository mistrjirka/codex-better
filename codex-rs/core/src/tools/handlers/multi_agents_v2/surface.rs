use super::followup_task::Handler as FollowupTaskHandler;
use super::interrupt_agent::Handler as InterruptAgentHandler;
use super::list_agents::Handler as ListAgentsHandler;
use super::send_message::Handler as SendMessageHandler;
use super::spawn::Handler as SpawnAgentHandler;
use super::wait::Handler as WaitAgentHandler;
use crate::session::session::Session;
use crate::tools::context::ToolInvocation;
use crate::tools::handlers::multi_agent_message::AgentMessageRoute;
use crate::tools::handlers::multi_agents_spec::SpawnAgentToolOptions;
use crate::tools::handlers::multi_agents_spec::WaitAgentTimeoutOptions;
use crate::tools::multi_agent_tool::multi_agent_v2_handler;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::NativeReview;
use crate::tools::registry::ToolRegistry;
use codex_prompts::ResolvedModelMessages;
use codex_tools::ResponsesApiNamespace;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ToolExecutor;
use codex_tools::ToolExposure;
use codex_tools::ToolName;
use codex_tools::ToolSearchInfo;
use codex_tools::ToolSpec;
use futures::future::BoxFuture;
use std::sync::Arc;

const CANONICAL_NAMESPACE_DESCRIPTION: &str = "Tools for spawning and managing sub-agents.";
pub(crate) const EXTERNAL_AGENT_NAMESPACE: &str = "external_agents";
const EXTERNAL_NAMESPACE_DESCRIPTION: &str =
    "Plaintext collaboration tools for authorized agents using non-OpenAI model providers.";

pub(crate) struct MultiAgentV2ToolOptions<'a> {
    pub(crate) namespace: Option<&'a str>,
    pub(crate) exposure: ToolExposure,
    pub(crate) provider_is_openai: bool,
    pub(crate) external_bridge_enabled: bool,
    pub(crate) disable_direct_message: bool,
    pub(crate) native_review: NativeReview,
    pub(crate) model_messages: &'a ResolvedModelMessages<'a>,
    pub(crate) spawn: SpawnAgentToolOptions,
    pub(crate) wait_agent: Option<WaitAgentTimeoutOptions>,
}

pub(crate) fn register_multi_agent_v2_tools(
    registry: &mut ToolRegistry,
    options: MultiAgentV2ToolOptions<'_>,
) {
    let MultiAgentV2ToolOptions {
        namespace,
        exposure,
        provider_is_openai,
        external_bridge_enabled,
        disable_direct_message,
        native_review,
        model_messages,
        spawn,
        wait_agent,
    } = options;
    let external_spawn =
        (provider_is_openai && external_bridge_enabled).then(|| SpawnAgentToolOptions {
            hide_agent_type_model_reasoning: spawn.hide_agent_type_model_reasoning,
            multi_agent_version: spawn.multi_agent_version,
            ..Default::default()
        });
    MessagingSurface::canonical(namespace, exposure, provider_is_openai).register(
        registry,
        spawn,
        Some(model_messages),
        if provider_is_openai {
            native_review
        } else {
            NativeReview::Disabled
        },
        disable_direct_message,
    );

    if let Some(external_spawn) = external_spawn {
        MessagingSurface::external().register(
            registry,
            external_spawn,
            /*model_messages*/ None,
            NativeReview::Disabled,
            disable_direct_message,
        );
    }
    if let Some(wait_agent) = wait_agent {
        registry.register_trusted_with_exposure(
            with_model_catalog_overrides(
                WaitAgentHandler::new(wait_agent),
                namespace,
                CANONICAL_NAMESPACE_DESCRIPTION,
                Some(model_messages),
                "wait_agent",
                /*is_spawn*/ false,
            ),
            exposure,
        );
    }
    registry.register_trusted_with_native_review(
        with_model_catalog_overrides(
            InterruptAgentHandler,
            namespace,
            CANONICAL_NAMESPACE_DESCRIPTION,
            Some(model_messages),
            "interrupt_agent",
            false,
        ),
        exposure,
        native_review,
    );
    registry.register_trusted_with_exposure(
        with_model_catalog_overrides(
            ListAgentsHandler,
            namespace,
            CANONICAL_NAMESPACE_DESCRIPTION,
            Some(model_messages),
            "list_agents",
            false,
        ),
        exposure,
    );
}

struct MessagingSurface {
    namespace: Option<String>,
    namespace_description: &'static str,
    exposure: ToolExposure,
    message_route: AgentMessageRoute,
}

impl MessagingSurface {
    fn canonical(
        namespace: Option<&str>,
        exposure: ToolExposure,
        provider_is_openai: bool,
    ) -> Self {
        let message_route = if provider_is_openai {
            AgentMessageRoute::Native
        } else {
            AgentMessageRoute::ProviderPlaintext
        };
        Self {
            namespace: namespace.map(str::to_string),
            namespace_description: CANONICAL_NAMESPACE_DESCRIPTION,
            exposure: message_route.messaging_tool_exposure(exposure),
            message_route,
        }
    }

    fn external() -> Self {
        Self {
            namespace: Some(EXTERNAL_AGENT_NAMESPACE.to_string()),
            namespace_description: EXTERNAL_NAMESPACE_DESCRIPTION,
            exposure: ToolExposure::DirectModelOnly,
            message_route: AgentMessageRoute::ExternalPlaintext,
        }
    }

    fn register(
        self,
        registry: &mut ToolRegistry,
        spawn_options: SpawnAgentToolOptions,
        model_messages: Option<&ResolvedModelMessages>,
        native_review: NativeReview,
        disable_direct_message: bool,
    ) {
        let Self {
            namespace,
            namespace_description,
            exposure,
            message_route,
        } = self;
        let namespace = namespace.as_deref();
        let spawn_description = model_messages
            .and_then(|messages| messages.multi_agent_tool_description_override("spawn_agent"))
            .map(str::to_owned);
        registry.register_trusted_with_native_review(
            with_model_catalog_overrides(
                SpawnAgentHandler::new(spawn_options, message_route, spawn_description),
                namespace,
                namespace_description,
                model_messages,
                "spawn_agent",
                /*is_spawn*/ true,
            ),
            exposure,
            native_review,
        );
        if !disable_direct_message {
            registry.register_trusted_with_native_review(
                with_model_catalog_overrides(
                    SendMessageHandler::new(message_route),
                    namespace,
                    namespace_description,
                    model_messages,
                    "send_message",
                    false,
                ),
                exposure,
                native_review,
            );
            registry.register_trusted_with_native_review(
                with_model_catalog_overrides(
                    FollowupTaskHandler::new(message_route),
                    namespace,
                    namespace_description,
                    model_messages,
                    "followup_task",
                    false,
                ),
                exposure,
                native_review,
            );
        }
    }
}

fn with_model_catalog_overrides(
    handler: impl CoreToolRuntime + 'static,
    namespace: Option<&str>,
    namespace_description: &str,
    messages: Option<&ResolvedModelMessages>,
    tool_name: &str,
    is_spawn: bool,
) -> Arc<dyn CoreToolRuntime> {
    if let Some(messages) = messages {
        multi_agent_v2_handler(
            handler,
            namespace,
            (!is_spawn)
                .then(|| messages.multi_agent_tool_description_override(tool_name))
                .flatten(),
            messages.multi_agent_tool_parameters_override(tool_name),
        )
    } else {
        // The authorized external bridge is a separate namespace, never a catalog alias.
        namespaced_handler_with_description(handler, namespace, namespace_description)
    }
}

fn namespaced_handler(
    handler: impl CoreToolRuntime + 'static,
    namespace: Option<&str>,
) -> Arc<dyn CoreToolRuntime> {
    namespaced_handler_with_description(handler, namespace, CANONICAL_NAMESPACE_DESCRIPTION)
}

fn namespaced_handler_with_description(
    handler: impl CoreToolRuntime + 'static,
    namespace: Option<&str>,
    namespace_description: &str,
) -> Arc<dyn CoreToolRuntime> {
    match namespace {
        Some(namespace) => Arc::new(NamespaceOverride {
            handler: Arc::new(handler),
            namespace: namespace.to_string(),
            namespace_description: namespace_description.to_string(),
        }),
        None => Arc::new(handler),
    }
}

struct NamespaceOverride {
    handler: Arc<dyn CoreToolRuntime>,
    namespace: String,
    namespace_description: String,
}

impl ToolExecutor<ToolInvocation> for NamespaceOverride {
    fn tool_name(&self) -> ToolName {
        ToolName::namespaced(self.namespace.clone(), self.handler.tool_name().name)
    }

    fn spec(&self) -> ToolSpec {
        match self.handler.spec() {
            ToolSpec::Function(tool) => ToolSpec::Namespace(ResponsesApiNamespace {
                name: self.namespace.clone(),
                description: self.namespace_description.clone(),
                tools: vec![ResponsesApiNamespaceTool::Function(tool)],
            }),
            spec => spec,
        }
    }

    fn exposure(&self) -> ToolExposure {
        self.handler.exposure()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        self.handler.supports_parallel_tool_calls()
    }

    fn search_info(&self) -> Option<ToolSearchInfo> {
        self.handler.search_info()
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        self.handler.handle(invocation)
    }
}

impl CoreToolRuntime for NamespaceOverride {
    fn direct_tool_call_source_policy(&self) -> crate::tools::context::DirectToolCallSourcePolicy {
        self.handler.direct_tool_call_source_policy()
    }

    fn wait_until_ready<'a>(&'a self, session: &'a Arc<Session>) -> Option<BoxFuture<'a, ()>> {
        self.handler.wait_until_ready(session)
    }

    fn matches_kind(&self, payload: &crate::tools::context::ToolPayload) -> bool {
        self.handler.matches_kind(payload)
    }

    fn create_diff_consumer(
        &self,
    ) -> Option<Box<dyn crate::tools::registry::ToolArgumentDiffConsumer>> {
        self.handler.create_diff_consumer()
    }
}
