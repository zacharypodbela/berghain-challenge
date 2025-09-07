from typing import Any

from django.contrib import admin, messages
from django.contrib.contenttypes.models import ContentType
from django.db.models import QuerySet
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .constants import CAPACITY, REJECTION_LIMIT
from .models import Game, LocalGame, Person, RemoteGame


class GameTypeListFilter(admin.SimpleListFilter):
    title = "Type"
    parameter_name = "game_type"

    def lookups(
        self, request: HttpRequest, model_admin: admin.ModelAdmin
    ) -> list[tuple[str, str]]:
        return [("remote", "Remote"), ("local", "Local")]

    def queryset(
        self, request: HttpRequest, queryset: QuerySet[Game]
    ) -> QuerySet[Game]:
        value = self.value()
        if value == "remote":
            ctype = ContentType.objects.get_for_model(RemoteGame)
            return queryset.filter(polymorphic_ctype=ctype)
        if value == "local":
            ctype = ContentType.objects.get_for_model(LocalGame)
            return queryset.filter(polymorphic_ctype=ctype)
        return queryset


@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = [
        "game_id_display",
        "view_details",
        "scenario",
        "status",
        "admitted_count",
        "rejected_count",
        "pending_count",
        "view_people",
        "view_on_challenge_site",
        "created_at",
    ]
    list_filter = ["scenario", "status", GameTypeListFilter, "created_at"]
    search_fields = ["game_id"]
    readonly_fields = [
        "created_at",
        "completed_at",
        "completion_reason",
        "admitted_count",
        "rejected_count",
        "pending_count",
    ]
    ordering = ["-created_at"]
    actions = [
        "start_scenario_1",
        "start_scenario_2",
        "start_scenario_3",
        "start_local_scenario_1",
        "start_local_scenario_2",
        "start_local_scenario_3",
    ]

    def game_id_display(self, obj: Game) -> SafeString | str:
        """Display game ID with type indicator"""
        if isinstance(obj, LocalGame):
            return format_html(
                '<span style="color: #0066cc;">[LOCAL]</span> {}', obj.game_id
            )
        elif isinstance(obj, RemoteGame):
            return format_html(
                '<span style="color: #00aa00;">[REMOTE]</span> {}', obj.game_id
            )
        return obj.game_id

    game_id_display.short_description = "Game ID"  # type: ignore [attr-defined]

    def start_scenario_1(self, request: HttpRequest, queryset: QuerySet[Game]) -> None:
        """Admin action to start a new scenario 1 game"""
        self._start_new_game(
            request,
            1,
            RemoteGame,
        )

    start_scenario_1.short_description = "Start new Scenario 1 game"  # type: ignore [attr-defined]

    def start_scenario_2(self, request: HttpRequest, queryset: QuerySet[Game]) -> None:
        """Admin action to start a new scenario 2 game"""
        self._start_new_game(
            request,
            2,
            RemoteGame,
        )

    start_scenario_2.short_description = "Start new Scenario 2 game"  # type: ignore [attr-defined]

    def start_scenario_3(self, request: HttpRequest, queryset: QuerySet[Game]) -> None:
        """Admin action to start a new scenario 3 game"""
        self._start_new_game(
            request,
            3,
            RemoteGame,
        )

    start_scenario_3.short_description = "Start new Scenario 3 game"  # type: ignore [attr-defined]

    def start_local_scenario_1(
        self, request: HttpRequest, queryset: QuerySet[Game]
    ) -> None:
        """Admin action to start a new local scenario 1 game"""
        self._start_new_game(request, 1, LocalGame)

    start_local_scenario_1.short_description = "Start new LOCAL Scenario 1 game"  # type: ignore [attr-defined]

    def start_local_scenario_2(
        self, request: HttpRequest, queryset: QuerySet[Game]
    ) -> None:
        """Admin action to start a new local scenario 2 game"""
        self._start_new_game(request, 2, LocalGame)

    start_local_scenario_2.short_description = "Start new LOCAL Scenario 2 game"  # type: ignore [attr-defined]

    def start_local_scenario_3(
        self, request: HttpRequest, queryset: QuerySet[Game]
    ) -> None:
        """Admin action to start a new local scenario 3 game"""
        self._start_new_game(request, 3, LocalGame)

    start_local_scenario_3.short_description = "Start new LOCAL Scenario 3 game"  # type: ignore [attr-defined]

    def _start_new_game(
        self, request: HttpRequest, scenario: int, game_class: type[Game]
    ) -> None:
        """Helper method to start a new remote game and show appropriate messages"""
        try:
            game = game_class.start_new_game(scenario)
            messages.success(
                request,
                f"Successfully started new Scenario {scenario} game: {game.game_id}",
            )
            messages.info(
                request,
                f"First person ready for decision. "
                f"Pending: {game.pending_count}, Admitted: {game.admitted_count}, Rejected: {game.rejected_count}",
            )
        except Exception as e:
            messages.error(
                request, f"Failed to start Scenario {scenario} game: {str(e)}"
            )

    def view_people(self, obj: Game) -> SafeString:
        """Create a link to view people for this specific game"""
        url = reverse("admin:bouncer_person_changelist") + f"?game__id__exact={obj.id}"
        return format_html('<a href="{}">View People ({})</a>', url, obj.people.count())

    view_people.short_description = "People"  # type: ignore [attr-defined]

    def view_on_challenge_site(self, obj: Game) -> str | SafeString:
        """Create a link to view this game on the challenge site"""
        if isinstance(obj, LocalGame):
            return "N/A"
        url = f"https://berghain.challenges.listenlabs.ai/game/{obj.game_id}"
        return format_html('<a href="{}" target="_blank">View Live Game</a>', url)

    view_on_challenge_site.short_description = "Challenge Site"  # type: ignore [attr-defined]

    def view_details(self, obj: Game) -> SafeString:
        """Create a link to the detailed game view"""
        url = reverse("admin:bouncer_game_detail", args=[obj.pk])
        return format_html('<a href="{}">View Details</a>', url)

    view_details.short_description = "Details"  # type: ignore [attr-defined]

    def has_add_permission(self, request: HttpRequest) -> bool:
        """Disable manual game creation - games should be created via start_new_game"""
        return False

    def get_urls(self) -> list[Any]:
        """Add custom URLs for detailed game views"""
        urls = super().get_urls()
        custom_urls = [
            path(
                "<path:object_id>/detail/",
                self.admin_site.admin_view(self.game_detail_view),
                name="bouncer_game_detail",
            ),
        ]
        return custom_urls + urls  # type: ignore [no-any-return]

    def game_detail_view(self, request: HttpRequest, object_id: str) -> HttpResponse:
        """Custom detailed view for a game showing comprehensive statistics"""
        game = get_object_or_404(Game, pk=object_id)

        # Calculate constraint progress
        constraint_progress = []
        for constraint in game.constraints:
            attr_name = constraint["attribute"]
            required = constraint["minCount"]

            # Count admitted people with this attribute
            actual = game.people.filter(
                decision=True, **{f"attributes__{attr_name}": True}
            ).count()

            percentage = (actual / required) * 100 if required > 0 else 0
            over_target = max(0, actual - required)

            constraint_progress.append(
                {
                    "name": attr_name.replace("_", " ").title(),
                    "actual": actual,
                    "required": required,
                    "percentage": percentage,
                    "over_target": over_target,
                    "is_met": actual >= required,
                }
            )

        # Calculate efficiency (acceptance rate)
        total_decisions = game.admitted_count + game.rejected_count
        efficiency = (
            (game.admitted_count / total_decisions * 100) if total_decisions > 0 else 0
        )

        # Calculate capacity info
        capacity_percentage = (game.admitted_count / CAPACITY) * 100

        # Rejection limit info
        rejections_until_limit = REJECTION_LIMIT - game.rejected_count

        # Time calculations
        duration = None
        if game.completed_at and game.created_at:
            delta = game.completed_at - game.created_at
            total_secs = int(max(0.0, delta.total_seconds()))
            hours = total_secs // 3600
            minutes = (total_secs % 3600) // 60
            if hours > 0:
                duration = f"{int(hours)}h {int(minutes)}m"
            else:
                duration = f"{int(minutes)}m"

        context = {
            **self.admin_site.each_context(request),
            "title": f"Game Details: {game.game_id}",
            "game": game,
            "constraint_progress": constraint_progress,
            "efficiency": efficiency,
            "capacity": CAPACITY,
            "capacity_percentage": capacity_percentage,
            "rejection_limit": REJECTION_LIMIT,
            "rejections_until_limit": rejections_until_limit,
            "duration": duration,
            "opts": self.model._meta,
        }

        return TemplateResponse(request, "admin/bouncer/game/detail.html", context)

    def changelist_view(
        self, request: HttpRequest, extra_context: dict[str, str] | None = None
    ) -> TemplateResponse:
        """Override changelist to add detail links"""
        if extra_context is None:
            extra_context = {}
        extra_context["show_detail_link"] = "true"  # String value for template
        return super().changelist_view(request, extra_context)

    class Media:
        css = {"all": ("admin/css/auto_width_columns.css",)}


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ["attributes_summary", "person_index", "game", "created_at"]
    list_filter = ["decision", "created_at"]
    search_fields = ["game__game_id"]
    readonly_fields = ["created_at"]
    ordering = ["game", "person_index"]
    actions = ["accept_person", "reject_person"]

    def attributes_summary(self, obj: Person) -> str:
        """Display a summary of the person's attributes"""
        return ", ".join(key for key, value in obj.attributes.items() if value)

    attributes_summary.short_description = "Attributes"  # type: ignore [attr-defined]

    def accept_person(self, request: HttpRequest, queryset: QuerySet[Person]) -> None:
        """Admin action to accept selected pending people"""
        accepted = 0
        errors = 0

        for person in queryset.filter(decision__isnull=True):
            try:
                person.make_decision(accept=True)
                accepted += 1
            except Exception as e:
                errors += 1
                messages.error(
                    request, f"Failed to accept Person {person.person_index}: {str(e)}"
                )

        if accepted:
            messages.success(request, f"Successfully accepted {accepted} people")
        if errors:
            messages.warning(request, f"{errors} people could not be processed")

    accept_person.short_description = "Accept selected pending people"  # type: ignore [attr-defined]

    def reject_person(self, request: HttpRequest, queryset: QuerySet[Person]) -> None:
        """Admin action to reject selected pending people"""
        rejected = 0
        errors = 0

        for person in queryset.filter(decision__isnull=True):
            try:
                person.make_decision(accept=False)
                rejected += 1
            except Exception as e:
                errors += 1
                messages.error(
                    request, f"Failed to reject Person {person.person_index}: {str(e)}"
                )

        if rejected:
            messages.success(request, f"Successfully rejected {rejected} people")
        if errors:
            messages.warning(request, f"{errors} people could not be processed")

    reject_person.short_description = "Reject selected pending people"  # type: ignore [attr-defined]

    class Media:
        css = {"all": ("admin/css/auto_width_columns.css",)}
