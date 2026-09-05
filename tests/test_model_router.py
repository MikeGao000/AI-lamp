import unittest

from lamp_core.autonomous_contracts import Language, PrivacyMode
from lamp_core.model_router import (
    ModelEndpoint,
    ModelRouter,
    ModelTask,
    NetworkHealth,
    RouteStatus,
    RoutingRequest,
    SessionBudget,
)


class ModelRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = ModelRouter(
            [
                ModelEndpoint(
                    provider="cloud",
                    model="generic-chat",
                    task=ModelTask.CHAT,
                    language=None,
                    is_local=False,
                    estimated_latency_ms=300,
                    priority=20,
                ),
                ModelEndpoint(
                    provider="cloud",
                    model="danish-chat",
                    task=ModelTask.CHAT,
                    language=Language.DA_DK,
                    is_local=False,
                    estimated_latency_ms=450,
                    priority=50,
                ),
                ModelEndpoint(
                    provider="local",
                    model="small-da-chat",
                    task=ModelTask.CHAT,
                    language=Language.DA_DK,
                    is_local=True,
                    estimated_latency_ms=900,
                    priority=100,
                ),
                ModelEndpoint(
                    provider="cloud",
                    model="vision-reader",
                    task=ModelTask.READING_VISION,
                    language=Language.ZH_CN,
                    is_local=False,
                    supports_vision=True,
                    estimated_latency_ms=700,
                ),
            ]
        )

    def test_exact_language_beats_a_generic_provider(self):
        route = self.router.route(
            RoutingRequest(ModelTask.CHAT, Language.DA_DK, PrivacyMode.CLOUD_ALLOWED)
        )
        self.assertEqual(route.status, RouteStatus.MODEL)
        self.assertEqual(route.endpoint.model, "danish-chat")

    def test_local_only_never_selects_cloud_endpoint(self):
        route = self.router.route(
            RoutingRequest(ModelTask.CHAT, Language.DA_DK, PrivacyMode.LOCAL_ONLY)
        )
        self.assertEqual(route.status, RouteStatus.MODEL)
        self.assertTrue(route.endpoint.is_local)
        self.assertEqual(route.endpoint.model, "small-da-chat")

    def test_local_command_never_selects_a_model(self):
        route = self.router.route(
            RoutingRequest(ModelTask.LOCAL_COMMAND, Language.EN, PrivacyMode.CLOUD_ALLOWED)
        )
        self.assertEqual(route.status, RouteStatus.LOCAL)
        self.assertIsNone(route.endpoint)

    def test_visual_requirement_rejects_text_only_endpoints(self):
        route = self.router.route(
            RoutingRequest(
                ModelTask.READING_VISION,
                Language.ZH_CN,
                PrivacyMode.CLOUD_ALLOWED,
                requires_vision=True,
                max_latency_ms=600,
            )
        )
        self.assertEqual(route.status, RouteStatus.UNAVAILABLE)
        self.assertIsNone(route.endpoint)

    def test_offline_or_exhausted_budget_uses_local_fallback_only(self):
        offline = self.router.route(
            RoutingRequest(
                ModelTask.CHAT,
                Language.DA_DK,
                PrivacyMode.CLOUD_ALLOWED,
                network_health=NetworkHealth.OFFLINE,
            )
        )
        self.assertEqual("small-da-chat", offline.endpoint.model)
        exhausted = self.router.route(
            RoutingRequest(
                ModelTask.CHAT,
                Language.DA_DK,
                PrivacyMode.CLOUD_ALLOWED,
                session_budget=SessionBudget(requests_remaining=0),
            )
        )
        self.assertEqual("small-da-chat", exhausted.endpoint.model)

    def test_degraded_network_prefers_local_and_exposes_fallbacks(self):
        route = self.router.route(
            RoutingRequest(
                ModelTask.CHAT,
                Language.DA_DK,
                PrivacyMode.CLOUD_ALLOWED,
                network_health=NetworkHealth.DEGRADED,
            )
        )
        self.assertEqual("small-da-chat", route.endpoint.model)
        self.assertEqual(("danish-chat", "generic-chat"), tuple(item.model for item in route.fallbacks))
