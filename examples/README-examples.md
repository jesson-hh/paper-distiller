# Examples

## Single-topic run, 5 papers

    paper-distiller --vault ~/MyVault --topic "diffusion models for finance" --n 5

## Author search

    paper-distiller --vault ~/MyVault --author "Huang Jian" --n 5

## Cheap exploration (N=1, no survey)

    paper-distiller --vault ~/MyVault --topic "neural ODE" --n 1

## Dry-run before committing API budget

    paper-distiller --vault ~/MyVault --topic "transport maps" --n 5 --dry-run

## Force re-distill an existing slug

    paper-distiller --vault ~/MyVault --topic "diffusion" --n 5 --force

## Override the model for one run

    paper-distiller --vault ~/MyVault --topic "X" --n 3 --model qwen-max

## Verbose mode (shows per-paper progress + tracebacks on error)

    paper-distiller --vault ~/MyVault --topic "X" --n 5 --verbose
