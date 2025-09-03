from django.contrib import admin
from django.contrib import messages
from django.urls import reverse
from django.utils.html import format_html
from .models import Game, Person


@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = [
        "game_id",
        "scenario",
        "status",
        "constraints_summary",
        "attribute_statistics_summary",
        "admitted_count",
        "rejected_count",
        "pending_count",
        "view_people",
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

    def constraints_summary(self, obj):
        """Display the game constraints in a readable format"""
        if not obj.constraints:
            return "No constraints"

        constraints_list = []
        for constraint in obj.constraints:
            attr_name = constraint.get("attribute", "Unknown")
            min_count = constraint.get("minCount", 0)
            constraints_list.append(f"{attr_name}: {min_count}")

        return ", ".join(constraints_list) if constraints_list else "No constraints"

    def attribute_statistics_summary(self, obj):
        """Display the game attribute statistics in a readable format"""
        if not obj.attribute_statistics:
            return "No statistics"

        freq = obj.attribute_statistics.get("relativeFrequencies", {})
        correlations = obj.attribute_statistics.get("correlations", {})

        if not freq:
            return "No frequency data"

        # Start with "Total" frequencies
        freq_list = [f"{attr}: {pct:.1%}" for attr, pct in freq.items()]
        result = "Total - " + ", ".join(freq_list)

        # Add correlations
        for attr1, corr_dict in correlations.items():
            if corr_dict:
                corr_list = [
                    f"{attr2}: {corr:.1%}"
                    for attr2, corr in corr_dict.items()
                    if corr != 1.0
                ]
                if corr_list:
                    result += f"<br>{attr1} - " + ", ".join(corr_list)

        return format_html(result)

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

    def view_people(self, obj):
        """Create a link to view people for this specific game"""
        url = reverse("admin:bouncer_person_changelist") + f"?game__id__exact={obj.id}"
        return format_html('<a href="{}">View People ({})</a>', url, obj.people.count())

    view_people.short_description = "People"

    def has_add_permission(self, request):
        """Disable manual game creation - games should be created via start_new_game"""
        return False

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

    def attributes_summary(self, obj):
        """Display a summary of the person's attributes"""
        return ", ".join(key for key, value in obj.attributes.items() if value)

    attributes_summary.short_description = "Attributes"

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
    
    class Media:
        css = {"all": ("admin/css/auto_width_columns.css",)}
