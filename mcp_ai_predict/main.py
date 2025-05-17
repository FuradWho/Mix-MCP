import logging
import click
from mcp_app import mcp_app

logging.basicConfig(level=logging.DEBUG)

@click.command()
@click.option("--port", default=8001, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    if transport == "sse":
        mcp_app.run("sse", port=port)
    else:
        mcp_app.run("stdio")
    return 0

if __name__ == "__main__":
    main()
