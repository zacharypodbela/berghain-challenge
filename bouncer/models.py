from django.db import models
from django.utils import timezone
import requests

PLAYER_ID = "5465bb54-f27c-48b7-9655-db22dc55a78b"
API_BASE_URL = "https://berghain.challenges.listenlabs.ai"


class Game(models.Model):
    STATUS_CHOICES = [
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    game_id = models.CharField(max_length=255, unique=True)
    scenario = models.IntegerField()
    constraints = models.JSONField()
    attribute_statistics = models.JSONField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="running")
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

    @classmethod
    def start_new_game(cls, scenario):
        """
        Start a new game by calling the API and storing the game and first person in the database.

        Args:
            scenario (int): Scenario number (1, 2, or 3)

        Returns:
            Game: The newly created game instance with first person

        Raises:
            ValueError: If scenario is not 1, 2, or 3
            requests.RequestException: If API calls fail
        """
        if scenario not in [1, 2, 3]:
            raise ValueError("Scenario must be 1, 2, or 3")

        # Step 1: Call new-game API
        new_game_url = f"{API_BASE_URL}/new-game"
        new_game_params = {"scenario": scenario, "playerId": PLAYER_ID}

        try:
            response = requests.get(new_game_url, params=new_game_params)
            response.raise_for_status()
            game_data = response.json()

            # Step 2: Create game in database if API call successful
            game = cls.objects.create(
                game_id=game_data["gameId"],
                scenario=scenario,
                constraints=game_data["constraints"],
                attribute_statistics=game_data["attributeStatistics"],
                status="running",
            )

            # Step 3: Get first person
            first_person_url = f"{API_BASE_URL}/decide-and-next"
            first_person_params = {"gameId": game.game_id, "personIndex": 0}

            first_person_response = requests.get(
                first_person_url, params=first_person_params
            )
            first_person_response.raise_for_status()
            first_person_data = first_person_response.json()

            # Step 4: Create first person in database
            # The API returns the next person to make a decision on (person #1)
            if "nextPerson" in first_person_data and first_person_data["nextPerson"]:
                person_data = first_person_data["nextPerson"]
                Person.objects.create(
                    game=game,
                    person_index=person_data["personIndex"],
                    attributes=person_data["attributes"],
                    decision=None,  # Pending
                )

            return game

        except requests.RequestException as e:
            # Clean up any created game if first person call fails
            if "game" in locals():
                game.delete()
            raise requests.RequestException(f"API call failed: {e}")

    def __str__(self):
        return f"Game {self.game_id} - Scenario {self.scenario} - Status: {self.status}"

    class Meta:
        ordering = ["-created_at"]


class Person(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="people")
    person_index = models.IntegerField()
    attributes = models.JSONField()
    decision = models.BooleanField(
        null=True, blank=True
    )  # True=accept, False=reject, None=pending
    created_at = models.DateTimeField(auto_now_add=True)

    def make_decision(self, accept):
        """
        Make a decision on this person by calling the API and updating the database.

        Args:
            accept (bool): True to accept the person, False to reject

        Returns:
            dict: API response data if successful, None if failed

        Raises:
            ValueError: If person already has a decision or game is not running
            requests.RequestException: If API call fails
        """
        if self.decision is not None:
            raise ValueError(
                f"Person {self.person_index} already has a decision: {self.decision}"
            )

        if self.game.status != "running":
            raise ValueError(
                f"Cannot make decisions on game with status: {self.game.status}"
            )

        # Call the API
        url = f"{API_BASE_URL}/decide-and-next"
        params = {
            "gameId": self.game.game_id,
            "personIndex": self.person_index,
            "accept": str(accept).lower(),
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            api_data = response.json()

            # Only update database if API call was successful
            self.decision = accept
            self.save(update_fields=["decision"])

            # Update game status if the game is completed or failed
            if api_data.get("status") == "completed":
                self.game.status = "completed"
                self.game.completed_at = timezone.now()
                self.game.save(update_fields=["status", "completed_at"])
            elif api_data.get("status") == "failed":
                self.game.status = "failed"
                # Store the failure reason in a way we can see it
                if "reason" in api_data:
                    # We could add a failure_reason field to Game model, but for now just save status
                    pass
                self.game.save(update_fields=["status"])

            # Store the next person if provided in the response
            if "nextPerson" in api_data and api_data["nextPerson"]:
                next_person_data = api_data["nextPerson"]
                # Only create if this person doesn't already exist
                if not Person.objects.filter(
                    game=self.game, person_index=next_person_data["personIndex"]
                ).exists():
                    Person.objects.create(
                        game=self.game,
                        person_index=next_person_data["personIndex"],
                        attributes=next_person_data["attributes"],
                        decision=None,  # Pending
                    )

            return api_data

        except requests.RequestException as e:
            raise requests.RequestException(f"API call failed: {e}")

    def __str__(self):
        decision_str = (
            "Accepted"
            if self.decision is True
            else "Rejected"
            if self.decision is False
            else "Pending"
        )
        return (
            f"Person {self.person_index} in Game {self.game.game_id} - {decision_str}"
        )

    class Meta:
        ordering = ["person_index"]
        unique_together = ["game", "person_index"]
