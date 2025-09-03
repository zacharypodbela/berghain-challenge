"""
Django management command to export person records for a game to CSV
"""

import csv
from django.core.management.base import BaseCommand, CommandError
from bouncer.models import Game, Person


class Command(BaseCommand):
    help = 'Export person records for a game to CSV with expanded attributes'

    def add_arguments(self, parser):
        parser.add_argument(
            'game_id',
            type=str,
            help='Game ID (UUID format) to export'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output CSV file path (default: game_<game_id>.csv)'
        )

    def handle(self, *args, **options):
        game_id = options['game_id']
        output_file = options.get('output') or f'game_{game_id}.csv'

        # Get and validate game
        try:
            game = Game.objects.get(game_id=game_id)
        except Game.DoesNotExist:
            raise CommandError(f'Game "{game_id}" does not exist')

        # Get all people for this game
        people = game.people.all().order_by('person_index')
        
        if not people.exists():
            raise CommandError(f'No people found for game "{game_id}"')

        self.stdout.write(f'Exporting {people.count()} people for game {game_id}...')

        # Collect all unique attribute keys across all people
        all_attributes = set()
        for person in people:
            if person.attributes:
                all_attributes.update(person.attributes.keys())
        
        # Sort attributes for consistent column ordering
        sorted_attributes = sorted(all_attributes)

        # Create CSV headers
        headers = [
            'Person Index',
            'Decision',
            'Decision Text',
            'Created At'
        ]
        
        # Add attribute columns with friendly names
        for attr in sorted_attributes:
            # Convert snake_case to Title Case
            friendly_name = attr.replace('_', ' ').title()
            headers.append(f'Attribute: {friendly_name}')

        # Write CSV file
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header row
                writer.writerow(headers)
                
                # Write data rows
                for person in people:
                    # Convert decision to readable text
                    if person.decision is True:
                        decision_text = 'Accepted'
                    elif person.decision is False:
                        decision_text = 'Rejected'
                    else:
                        decision_text = 'Pending'
                    
                    # Basic person data
                    row = [
                        person.person_index,
                        person.decision,
                        decision_text,
                        person.created_at.isoformat()
                    ]
                    
                    # Add attribute values (True/False)
                    for attr in sorted_attributes:
                        value = person.attributes.get(attr, False) if person.attributes else False
                        row.append(value)
                    
                    writer.writerow(row)

            self.stdout.write(
                self.style.SUCCESS(f'Successfully exported to {output_file}')
            )
            
            # Show summary
            self.stdout.write(
                f'Game: {game.game_id} (Scenario {game.scenario})\n'
                f'Status: {game.status}\n'
                f'People exported: {people.count()}\n'
                f'Attributes found: {", ".join(sorted_attributes)}\n'
                f'File: {output_file}'
            )

        except IOError as e:
            raise CommandError(f'Error writing to file {output_file}: {e}')