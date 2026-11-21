# Terminal client

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

The user approved a full-screen terminal interface for human players. Use Python's
standard-library curses rather than adding a UI dependency. A numbered prompt was
considered but is less convenient for seeing all nine stones while choosing moves.

Provide `python -m schotten.terminal create` and `join GAME_ID`, with a server URL
option. Display both sides of all stones, ownership, the hand, game phase, counts,
and results. Arrow keys select cards and stones; Enter submits a legal play and
all locally provable claims, including claims enabled by that play. Pass only
when the server reports no legal plays. Highlight legal targets using the server's
legal-moves response. Display stone numbers as 1–9, converting to API indices.

Perform network work outside the curses event loop so quitting and resizing work
while waiting for an opponent or server. Recover from rejected turns by refreshing
state, display connection errors, and never automatically retry uncertain writes.
The server remains authoritative. Restore the terminal on exit. Do not change
server rules or authentication. Support terminals of at least 80 columns by 24
rows, with a resize prompt for smaller windows and color-independent card labels.
