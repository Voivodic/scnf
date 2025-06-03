{
    description = "Flake for the creation of a shell with SCNF";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        gitpkgs.url = "github:Voivodic/nix-derivations";
        gitpkgs.inputs.nixpkgs.follows = "nixpkgs";

    };

    outputs = { self, nixpkgs, gitpkgs, ... } @ inputs: 
    let
        # Set the system and the pkgs used
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; };
        git-pkgs = import gitpkgs { inherit pkgs; };

        # Install SCNF
        scnf =  pkgs.python313Packages.buildPythonPackage {
            pname = "SCNF";
            version = "0.1.0";
            format = "pyproject";

            src = ./.;

            propagatedBuildInputs = [
                pkgs.python313
                pkgs.python313Packages.tqdm
                pkgs.python313Packages.jaxtyping
                pkgs.python313Packages.optax
                git-pkgs.python313.e3nn
                git-pkgs.python313.diffrax
            ];

            meta = {
                description = "Steerable continuous normalizing flows.";
                homepage = "https://github.com/Voivodic/SCNF";
            };
        };
    in
    {
        # Instructions for the creation of the shell
        devShells.${system}.default = pkgs.mkShell{
            buildInputs = [
                scnf
            ];

            shellHook = ''
            '';
        };
    };
}
