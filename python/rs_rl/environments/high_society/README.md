# High Society

This directory is reserved for the manually programmed High Society environment.

Planned concerns:

- Game state and rules.
- Legal actions and action validation.
- Player observations.
- Rewards and episode termination.
- Deterministic seeding.
- Optional text or visual rendering.
- Environment-level tests.

No implementation has been added yet.

## Implementation

### Starting Conditions

Each player starts with the set:
1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 25000

### Turns

Default Route
1. A new card is drawn (if card is bad follow alt route)
2. Starting player bids or sits out
3. Each following player bids higher or sits out
4. Once only 1 player is still bidding they pick up the card

Bad Card Route:
1. Every player is now bidding not to pick the card up
2. Each player bids >= previous until someone is not willing
3. That player picks up the card for free and everyone else pays