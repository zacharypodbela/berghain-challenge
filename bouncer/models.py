from django.db import models
import json

PLAYER_ID = '5465bb54-f27c-48b7-9655-db22dc55a78b'


class Game(models.Model):
    STATUS_CHOICES = [
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    game_id = models.CharField(max_length=255, unique=True)
    scenario = models.IntegerField()
    constraints = models.JSONField()
    attribute_statistics = models.JSONField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='running')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    @property
    def admitted_count(self):
        return self.people.filter(decision=True).count()
    
    @property
    def rejected_count(self):
        return self.people.filter(decision=False).count()
    
    @property
    def pending_count(self):
        return self.people.filter(decision__isnull=True).count()
    
    def __str__(self):
        return f"Game {self.game_id} - Scenario {self.scenario} - Status: {self.status}"
    
    class Meta:
        ordering = ['-created_at']


class Person(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='people')
    person_index = models.IntegerField()
    attributes = models.JSONField()
    decision = models.BooleanField(null=True, blank=True)  # True=accept, False=reject, None=pending
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        decision_str = "Accepted" if self.decision is True else "Rejected" if self.decision is False else "Pending"
        return f"Person {self.person_index} in Game {self.game.game_id} - {decision_str}"
    
    class Meta:
        ordering = ['person_index']
        unique_together = ['game', 'person_index']
