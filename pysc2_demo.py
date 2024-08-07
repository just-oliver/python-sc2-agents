#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:01:43 2024

@author: deck
"""

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

class PyAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(PyAgent, self).step(obs)
        
        if buildSupplyDepot(self, obs):
            # where will the supply depot be built
            x = random.randint(0,83)
            y = random.randint(0,83)
            return actions.FUNCTIONS.Build_SupplyDepot_screen('now', (x, y))
        
        if buildBarracks(self, obs):
            # where will the Barracks be built
            x = random.randint(0,83)
            y = random.randint(0,83)
            return actions.FUNCTIONS.Build_Barracks_screen('now', (x, y))
        
        if buildMarines(self, obs):
            return actions.FUNCTIONS.Train_Marine_quick('now')
        
        if attack(self, obs):
            return actions.FUNCTIONS.Attack_minimap(0, [19, 23])
        
        marines = [unit for unit in obs.observation['feature_units']
                   if unit.unit_type == units.Terran.Marine]
        
        if len(marines) > 5:
            marine = random.choice(marines)
            return actions.FUNCTIONS.select_point("select_all_type", (marine.x, marine.y))

        
        barracks = [unit for unit in obs.observation['feature_units']
                    if unit.unit_type == units.Terran.Barracks]
        
        if len(barracks) > 0:
            barrack = random.choice(barracks)
            return actions.FUNCTIONS.select_point("select_all_type", (barrack.x, barrack.y))
        
        drones = [unit for unit in obs.observation['feature_units']
                  if unit.unit_type == units.Terran.SCV]
        if len(drones) > 0:
            drone = random.choice(drones)
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
        return actions.FUNCTIONS.no_op()

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]
    
    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True
        
        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False
    
def buildMarines(self, obs):
    marine = self.get_units_by_type(obs, units.Terran.Marine)
    if len(marine) <= 4:
        if self.unit_type_is_selected(obs, units.Terran.Barracks):
            if (actions.FUNCTIONS.Train_Marine_quick.id in 
                obs.observation.available_actions):
                return True
            return False

def attack(self, obs):
    marine = self.get_units_by_type(obs, units.Terran.Marine)
    if len(marine) > 5:
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if (actions.FUNCTIONS.Attack_screen.id in
                    obs.observation.available_actions):
                return True
            return False

def buildSupplyDepot(self, obs):
    supplyDepots = self.get_units_by_type(obs, units.Terran.SupplyDepot)
    if len(supplyDepots) == 0:
        if self.unit_type_is_selected(obs, units.Terran.SCV):
            if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in
                    obs.observation.available_actions):
                return True
            return False

def buildBarracks(self, obs):
    barracks = self.get_units_by_type(obs, units.Terran.Barracks)
    if len(barracks) == 0:
        if self.unit_type_is_selected(obs, units.Terran.SCV):
            if (actions.FUNCTIONS.Build_Barracks_screen.id in
                    obs.observation.available_actions):
                return True
            return False


def main(unused_argv):
    agent = PyAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.terran),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                step_mul=8,
                game_steps_per_episode=0,
                visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                print(f"Action space: {env.action_spec()}")
                print(f"Observation space: {env.observation_spec()}")
                
                timesteps = env.reset()
                agent.reset()
                
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass
            

if __name__ == '__main__':
    app.run(main)