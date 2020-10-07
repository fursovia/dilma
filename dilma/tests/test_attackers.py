from pathlib import Path

from allennlp.common import Params

import advsber
from advsber.utils.data import load_jsonlines
from advsber.settings import TransactionsData


PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"


class TestTransactionAttackers:
    def test_from_params(self):
        data = load_jsonlines(str(PROJECT_ROOT / "presets/age/sample.jsonl"))

        for config_path in (CONFIG_DIR / "attackers").glob("*.jsonnet"):
            try:
                params = Params.from_file(
                    str(config_path),
                    ext_vars={
                        "DATA_PATH": "",
                        "OUTPUT_PATH": "",
                        "CLF_PATH": str(PROJECT_ROOT / "presets/age/models/target_clf/gru_target_age.tar.gz"),
                        "MASKED_LM_PATH": str(PROJECT_ROOT / "presets/age/models/lm/lm.model.tar.gz"),
                    },
                )
                params["attacker"]["device"] = -1
                attacker = advsber.Attacker.from_params(params["attacker"])
            except Exception as e:
                raise AssertionError(f"unable to load params from {config_path}, because {e}")

            output = attacker.attack(TransactionsData(**data[0]))
            assert isinstance(output, advsber.AttackerOutput)
            assert isinstance(output.wer, int)
            assert output.wer >= 0
            assert isinstance(output.prob_diff, float)
            assert abs(output.prob_diff) <= 1.0
            assert isinstance(output.probability, float)
            assert output.probability >= 0.0
            assert isinstance(output.adversarial_probability, float)
            assert output.adversarial_probability >= 0.0
