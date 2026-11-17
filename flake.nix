{
  description = "Python and CUDA Environment for LLM from Scratch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = { allowUnfree = true; };
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312
          python312Packages.pip
          python312Packages.virtualenv
          python312Packages.ipykernel
          python312Packages.numpy
          python312Packages.pandas
          python312Packages.matplotlib
          python312Packages.pyarrow
          python312Packages.pyyaml
          python312Packages.tiktoken
        ];

        shellHook = ''
          if [ ! -d ".venv" ]; then
            python -m venv .venv
            echo "Created virtual environment (.venv)"
          fi
          
          source .venv/bin/activate
          
          # Expose both the Nvidia drivers AND the Nix C++ standard library to pip wheels
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

          echo "Virtual environment ready."
        '';
      };
    };
}