from django.contrib import admin
from .models import Game, Person


@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = ['game_id', 'scenario', 'status', 'admitted_count', 'rejected_count', 'pending_count', 'created_at']
    list_filter = ['scenario', 'status', 'created_at']
    search_fields = ['game_id']
    readonly_fields = ['created_at', 'admitted_count', 'rejected_count', 'pending_count']
    ordering = ['-created_at']


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ['person_index', 'game', 'decision', 'created_at']
    list_filter = ['decision', 'created_at']
    search_fields = ['game__game_id']
    readonly_fields = ['created_at']
    ordering = ['game', 'person_index']
