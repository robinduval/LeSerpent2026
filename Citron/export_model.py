"""Exporte un checkpoint léger à distribuer avec le code dans le dépôt."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    destination = ROOT / 'model'
    destination.mkdir(exist_ok=True)
    output = destination / 'model.pt'
    data = dict(online=source['online'], features=source.get('features','basic'),
                steps=source['steps'], updates=source['updates'], inference_only=True)
    torch.save(data, output)
    (destination/'metadata.json').write_text(json.dumps(dict(
        source=str(args.checkpoint), features=data['features'], training_steps=data['steps'],
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        evaluation_clock_hz=5, selection_note='Checkpoint spatial ayant atteint 82 points lors de la partie interrompue seed 1000 ; score non homologué comme partie terminée.'),indent=2))
    print(output, output.stat().st_size, 'octets')


if __name__=='__main__':
    main()
