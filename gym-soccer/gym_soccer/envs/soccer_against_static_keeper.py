import logging
from gym_soccer.envs.soccer_empty_goal import SoccerEmptyGoalEnv

logger = logging.getLogger(__name__)

class SoccerAgainstStaticEnv(SoccerEmptyGoalEnv):
    """
    SoccerAgainstKeeper initializes the agent most of the way down the
    field with the ball and tasks it with scoring on a keeper.

    Rewards in this task are the same as SoccerEmptyGoal: reward
    is given for kicking the ball close to the goal and extra reward is
    given for scoring a goal.

    """
    def __init__(self):
        self._static_keeper = True #Added a flag to know if we start a static keeper
        super(SoccerAgainstStaticEnv, self).__init__()

    def _configure_environment(self):
        super(SoccerAgainstStaticEnv, self)._start_hfo_server(defense_agents = 1,#defense_agents=1, #defense_npcs = 1
                                                              offense_on_ball=0, #changed from 0 to 1
                                                              ball_x_min=0, ball_x_max=0.45)
