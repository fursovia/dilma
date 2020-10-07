from allennlp.common.params import Params
import typer
import jsonlines
from tqdm import tqdm

from dilma.attackers import Attacker
from dilma.common import SequenceData
from dilma.utils.data import load_jsonlines


def main(config_path: str, samples: int = typer.Option(None, help="Number of samples")):
    params = Params.from_file(config_path)
    attacker = Attacker.from_params(params["attacker"])

    data = load_jsonlines(params["data_path"])[:samples]

    output_path = params["output_path"]
    typer.echo(f"Saving results to {output_path}")
    with jsonlines.open(output_path, "w") as writer:
        for example in tqdm(data):
            adversarial_output = attacker.attack(SequenceData(**example))
            writer.write(adversarial_output.to_dict())


if __name__ == "__main__":
    typer.run(main)
