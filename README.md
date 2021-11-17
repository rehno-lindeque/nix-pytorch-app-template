# Nix PyTorch application templates

This template is currently for my own use, but you are welcome to try it. The code is currently still pretty rough and likely to evolve.

## Instantiating the templates

To create a simple pytorch app, use the following command.

```
nix flake init -t github:rehno-lindeque/nix-pytorch-app-template#pytorch-app
```

## Developing the templates

Run `nix develop` to get started. 

If you don't have nix 2.0: `nix-shell -p nixFlakes --run 'nix develop'`.

Make sure that `experimental-features = nix-command flakes` is turned on, since this is a flake.
