use std::{
    future::IntoFuture,
    net::{IpAddr, SocketAddr},
    process::ExitCode,
};

use clap::{Parser, Subcommand};
use schotten_native::{bots, server, tournament_cli};

#[derive(Parser)]
#[command(
    about = "Native Schotten Totten REST server and bot simulator",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Tournament(tournament_cli::Args),
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value_t = 0)]
        workers: usize,
        #[arg(long, default_value = "tournament-recordings")]
        recordings_dir: std::path::PathBuf,
    },
    Series {
        #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..))]
        games: u32,
        #[arg(long, default_value = "greedy", value_parser = bots::parse_strategy_arg)]
        strategy: String,
        #[arg(long, default_value = "random", value_parser = bots::parse_strategy_arg)]
        opponent_strategy: String,
        #[arg(long)]
        seed: Option<i64>,
        #[arg(long)]
        bot_seed: Option<i64>,
        #[arg(long, default_value_t = 0)]
        threads: usize,
        #[arg(short, long)]
        quiet: bool,
    },
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Tournament(args) => tournament_cli::execute(args),
        Command::Serve {
            host,
            port,
            workers,
            recordings_dir,
        } => {
            let workers = if workers == 0 {
                std::thread::available_parallelism().map_or(1, usize::from)
            } else {
                workers
            };
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(workers)
                .enable_all()
                .build()
                .map_err(|error| error.to_string())
                .and_then(|runtime| {
                    runtime.block_on(async {
                        let listener = tokio::net::TcpListener::bind(SocketAddr::new(host, port))
                            .await
                            .map_err(|error| error.to_string())?;
                        eprintln!(
                            "Schotten Totten listening on http://{}",
                            listener.local_addr().map_err(|error| error.to_string())?
                        );
                        let (shutdown,mut draining)=tokio::sync::watch::channel(false);
                        let app=server::router_with_options(shutdown.clone(), recordings_dir);
                        let serving=axum::serve(listener,app).with_graceful_shutdown(async move {let _=draining.changed().await;}).into_future();
                        tokio::pin!(serving);
                        tokio::select! {
                            result=&mut serving => result.map_err(|error| error.to_string()),
                            _=tokio::signal::ctrl_c() => {
                                shutdown.send_replace(true);
                                match tokio::time::timeout(std::time::Duration::from_secs(3),&mut serving).await {
                                    Ok(result)=>result.map_err(|error| error.to_string()),
                                    Err(_)=>Ok(()),
                                }
                            }
                        }
                    })
                })
        }
        Command::Series {
            games,
            strategy,
            opponent_strategy,
            seed,
            bot_seed,
            threads,
            quiet,
        } => bots::series(
            games as usize,
            &strategy,
            &opponent_strategy,
            seed,
            bot_seed,
            threads,
            quiet,
        ),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_every_native_strategy_and_rejects_unknown_names() {
        for name in bots::STRATEGIES {
            assert!(
                Cli::try_parse_from([
                    "schotten-native",
                    "series",
                    "--strategy",
                    name,
                    "--opponent-strategy",
                    name
                ])
                .is_ok()
            );
        }
        assert!(
            Cli::try_parse_from(["schotten-native", "series", "--strategy", "unknown"]).is_err()
        );
    }
}
