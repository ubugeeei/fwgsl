///|
/// `fwgsl` binary entry point. Drives `@cli.parse_args` over `@env.args()`
/// and dispatches to `@cli.run_command`.
///
/// File I/O for `<file>` inputs is not available on the `wasm-gc` target
/// the rest of the workspace builds against, so the binary surfaces a
/// clear "use --source or --stdin" error in that case rather than failing
/// silently. Host wrappers (a Node loader, the playground, or a future
/// native build) feed the source string in directly and call
/// `@cli.run_command` themselves.
fn main {
  let argv = drop_program_name(@env.args())
  let command = @cli.parse_args(argv)
  match command {
    @cli.CliCommand::Help => {
      println(@cli.help_text())
      return
    }
    @cli.CliCommand::Version => {
      println("fwgsl " + @cli.VERSION)
      return
    }
    @cli.CliCommand::Invalid(msg) => {
      println(msg)
      println("")
      println(@cli.help_text())
      return
    }
    _ => ()
  }
  match input_for(command) {
    @cli.CliInput::Inline(text) => {
      let (out, _) = @cli.run_command(command, text, "<inline>")
      println(out)
    }
    @cli.CliInput::File(path) => {
      println(
        "fwgsl: reading source files from `" +
        path +
        "` is not available on this build.",
      )
      println("       pipe source via `--stdin` or pass it inline with `--source <text>`.")
    }
    @cli.CliInput::Stdin =>
      println(
        "fwgsl: `--stdin` requires a host wrapper that feeds STDIN into `@cli.run_command`.",
      )
  }
}

///|
/// `@env.args()` includes the program name on every supported runtime;
/// drop it before handing the rest to the parser.
fn drop_program_name(argv : Array[String]) -> Array[String] {
  if argv.length() == 0 {
    return argv
  }
  let out : Array[String] = Array::new()
  let mut i = 1
  while i < argv.length() {
    out.push(argv[i])
    i = i + 1
  }
  out
}

///|
fn input_for(command : @cli.CliCommand) -> @cli.CliInput {
  match command {
    @cli.CliCommand::Compile(inp) => inp
    @cli.CliCommand::Check(inp) => inp
    @cli.CliCommand::Fmt(inp) => inp
    _ => @cli.CliInput::Inline("")
  }
}
