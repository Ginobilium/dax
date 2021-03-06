{ pkgs ? import <nixpkgs> {}, artiqpkgs ? import <artiq-full> { inherit pkgs; }, daxVersion ? null }:

with pkgs;
python3Packages.buildPythonPackage rec {
  pname = "dax";
  version = if daxVersion == null then import ./version.nix { inherit pkgs src; } else daxVersion;

  src = nix-gitignore.gitignoreSource [ "*.nix" ] ./.;

  VERSIONEER_OVERRIDE = version;

  propagatedBuildInputs = (with artiqpkgs; [ artiq sipyco ])
    ++ (with python3Packages; [ numpy scipy pyvcd natsort pygit2 matplotlib graphviz h5py networkx ]);

  checkInputs = with python3Packages; [ pytestCheckHook ];

  inherit (python3Packages.pygit2) SSL_CERT_FILE;

  meta = with stdenv.lib; {
    description = "Duke ARTIQ Extensions (DAX)";
    maintainers = [ "Duke University" ];
    homepage = "https://gitlab.com/duke-artiq/dax";
    license = licenses.asl20;
  };
}
