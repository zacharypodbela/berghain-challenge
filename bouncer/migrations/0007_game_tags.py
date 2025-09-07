from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("bouncer", "0006_localgame_remotegame_game_polymorphic_ctype"),
    ]

    operations = [
        migrations.AddField(
            model_name="game",
            name="tags",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
