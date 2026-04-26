import unittest

from poker44.validator.sanitization import prepare_hand_for_miner


class SanitizationFocusSeatTests(unittest.TestCase):
    def test_preserves_hero_seat_for_canonical_eval_hands(self):
        payload = {
            "metadata": {
                "game_type": "Hold'em",
                "limit_type": "No Limit",
                "max_seats": 2,
                "hero_seat": 2,
                "hand_ended_on_street": "",
                "button_seat": 0,
                "sb": 0.01,
                "bb": 0.02,
                "ante": 0.0,
                "rng_seed_commitment": None,
            },
            "players": [
                {
                    "player_uid": "seat_1",
                    "seat": 1,
                    "starting_stack": 10.0,
                    "hole_cards": None,
                    "showed_hand": False,
                },
                {
                    "player_uid": "seat_2",
                    "seat": 2,
                    "starting_stack": 10.0,
                    "hole_cards": None,
                    "showed_hand": False,
                },
            ],
            "streets": [],
            "actions": [
                {
                    "action_id": "1",
                    "street": "preflop",
                    "actor_seat": 1,
                    "action_type": "call",
                    "amount": 0.1,
                    "raise_to": None,
                    "call_to": 0.1,
                    "normalized_amount_bb": 5.0,
                    "pot_before": 0.1,
                    "pot_after": 0.2,
                }
            ],
            "outcome": {
                "winners": [],
                "payouts": {},
                "total_pot": 0.2,
                "rake": 0.0,
                "result_reason": "fold",
                "showdown": False,
            },
        }

        prepared = prepare_hand_for_miner(payload)

        self.assertEqual(prepared["metadata"]["hero_seat"], 2)


if __name__ == "__main__":
    unittest.main()
