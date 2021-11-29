{ lib
, buildPythonApplication
, python
, pytorch
, kornia
, torchvision
, pandas
, setuptools
, tqdm
, wandb
}:


buildPythonApplication {
  pname    = "pytorch-app";
  version = "1.0";

  src = ./.;
  propagatedBuildInputs = [
    pytorch
    kornia
    setuptools
    torchvision
    tqdm
    wandb
  ];

  installPhase = ''
    mkdir -p $out $out/bin
    cp -r app $out/share

    chmod +x $out/share/train/train.py
    wrapPythonProgramsIn $out/share/train "$out/share/train $pythonPath"
    ln -s $out/share/train/train.py $out/bin/train

    chmod +x $out/share/inference/inference.py
    wrapPythonProgramsIn $out/share/inference "$out/share/inference $pythonPath"
    ln -s $out/share/inference/inference.py $out/bin/inference
  '';

}

