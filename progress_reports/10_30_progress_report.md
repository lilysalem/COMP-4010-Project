### Group 17 progress report - 30 Oct 2025

We've made good headway in the past 2 weeks. Our group was successful in our project demo, but there are some components
of the environment as is that are still WIP. 

#### Progress made in the last 2 weeks:

Over the past 2 weeks, we achieved the following:

- Implemented a demonstration environment that is capable of rendering to screen
- Created some variant agent classes for future multi-agent work
- Damon made signficant improvements to our initial coordinate adressing system
- Documented a small "getting started" guide for future use


#### Plans in the next 2 weeks:

In the next 2 weeks, we plan to:

- Refactor our world and agent classes to match OpenAI Gymnasium API specs more closely, more closely separating out
environment and agent responsibilities
- Redefine our state spaces and visible state for agents to allow for actual learning
- Create testing files and add a working reset function in terms of our environment
- Allow for storing and reusing generated environment presets
- Stretch goal - start experimenting with different learning approaches with single agents