{
  description = "Python and Jupyter Environment for LLM from Scratch";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          allowBroken = true;
        };
      };

      # Using Python 3.12 for maximum package compatibility
      myPython = pkgs.python312;

      pythonEnv = myPython.withPackages (ps: with ps; [
        ipykernel
        jupyter
        numpy
        pandas
        matplotlib
        torch
        tiktoken
        tensorflow
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pythonEnv
        ];

        shellHook = ''
          python -m ipykernel install --user --name=nix-llm-kernel --display-name "Python 3.12 (LLM Flake)" > /dev/null 2>&1
          echo "LLM development environment loaded successfully."
        '';
      };
    };
}
