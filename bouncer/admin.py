from django.contrib import admin
from django.contrib import messages
from .models import Game, Person


@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = [
        "game_id",
        "scenario",
        "status",
        "admitted_count",
        "rejected_count",
        "pending_count",
        "created_at",
    ]
    list_filter = ["scenario", "status", "created_at"]
    search_fields = ["game_id"]
    readonly_fields = [
        "created_at",
        "admitted_count",
        "rejected_count",
        "pending_count",
    ]
    ordering = ["-created_at"]
    actions = ["start_scenario_1", "start_scenario_2", "start_scenario_3"]

    def start_scenario_1(self, request, queryset):
        """Admin action to start a new scenario 1 game"""
        self._start_new_game(request, 1)

    start_scenario_1.short_description = "Start new Scenario 1 game"

    def start_scenario_2(self, request, queryset):
        """Admin action to start a new scenario 2 game"""
        self._start_new_game(request, 2)

    start_scenario_2.short_description = "Start new Scenario 2 game"

    def start_scenario_3(self, request, queryset):
        """Admin action to start a new scenario 3 game"""
        self._start_new_game(request, 3)

    start_scenario_3.short_description = "Start new Scenario 3 game"

    def _start_new_game(self, request, scenario):
        """Helper method to start a new game and show appropriate messages"""
        try:
            game = Game.start_new_game(scenario)
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

    def has_add_permission(self, request):
        """Disable manual game creation - games should be created via start_new_game"""
        return False


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ["person_index", "game", "decision", "created_at"]
    list_filter = ["decision", "created_at"]
    search_fields = ["game__game_id"]
    readonly_fields = ["created_at"]
    ordering = ["game", "person_index"]
    actions = ["accept_person", "reject_person"]

    def accept_person(self, request, queryset):
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

    accept_person.short_description = "Accept selected pending people"

    def reject_person(self, request, queryset):
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

    reject_person.short_description = "Reject selected pending people"
