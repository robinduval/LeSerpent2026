{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; cudaSupport = true; }; } }:

pkgs.mkShell {
  buildInputs = with pkgs.python3Packages; [
    pygame
    torchWithCuda
    numpy
  ];

  shellHook = ''
    echo "=========================================================="
    echo "🐍 Environnement Deep RL prêt ! RTX 3050 Parée au décollage."
    echo "=========================================================="
    python -c "import torch; print('-> Statut CUDA :', 'Actif sur ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'INACTIF (CPU)')"
  '';
}