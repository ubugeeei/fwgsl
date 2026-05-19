{
  description = "fwgsl — pure functional WebGPU language compiling to WGSL";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "fwgsl";

          # MoonBit is not yet in nixpkgs and ships its own installer that
          # drops binaries under ~/.moon/bin. The shell hook installs it on
          # first entry and adds the directory to PATH on every entry.
          packages = with pkgs; [
            nodejs_22
            python3       # playground dev server (`just playground`)
            just          # task runner that replaces the old mise tasks
            curl
            git
          ];

          shellHook = ''
            export PATH="$HOME/.moon/bin:$PATH"
            if ! command -v moon >/dev/null 2>&1; then
              echo "MoonBit not detected — installing into ~/.moon"
              curl -fsSL https://cli.moonbitlang.com/install/unix.sh | bash
              # Re-export PATH so the just-installed moon is reachable in
              # the same shell session.
              export PATH="$HOME/.moon/bin:$PATH"
            fi
            echo "fwgsl dev shell — try \`just --list\` for available tasks"
          '';
        };
      });
}
