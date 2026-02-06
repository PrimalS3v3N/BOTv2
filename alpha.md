# Alpha - Strategy Ideas & TODOs

## TODOs

- [x] Overbought/Oversold instant exit: When RSI >= 85 at entry for CALLs (overbought) or RSI <= 15 at entry for PUTs (oversold), buy the contract and immediately sell it. Exit reason recorded as "Overbought" or "Oversold". This goes against the check DD strategy. Reference: check DD function (to be implemented).
- [ ] Consider buying the opposite direction when overbought/oversold is detected (e.g. PUTs instead of CALLs when overbought). Reference: check DD function.
