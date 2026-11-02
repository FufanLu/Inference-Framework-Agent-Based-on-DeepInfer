fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = "../../proto";
    let proto = "../../proto/leviathan/worker/v1/worker.proto";
    println!("cargo:rerun-if-changed={proto}");
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&[proto], &[proto_root])?;
    Ok(())
}
