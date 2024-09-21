from __future__ import annotations

from typing import Any
import json
from pathlib import Path

from sweagent.agent.agents import Agent, AgentArguments, TrajectoryStep
from sweagent.agent.commands import Command
from sweagent.environment.swe_env import SWEEnv
from sweagent.utils.log import get_logger
from sweagent.agent.models import (
    APIStats,
)

class MAgent(Agent):
    def __init__(self, name: str, args: AgentArguments):
        super().__init__(name, args)

        #print("!! model="+str(self.model))
        #print("!! args="+str(args))
        #print("!! commands="+str(args.config._commands))
        #print("!! subroutine="+str(args.config.subroutine_types))

        tools = []
        for cmd in args.config._commands:
            tools.append(self.command2fncall(cmd))

        #print("!! tools = " + str(tools))

    def run(
        self,
        setup_args: dict[str, Any],
        env: SWEEnv,
        observation: str | None = None,
        traj_dir: Path | None = None,
        return_type: str | None = "info_trajectory",
        init_model_stats: APIStats | None = None,
    ):
        """
        Run the agent on an environment.
        Return the final value of the specified return type.

        Args:
            setup_args: Arguments to pass to the agent's setup method.
            env: The environment to run the agent on.
            observation: Output from environment setup
            traj_dir: Directory to save the trajectory to
            return_type: Controls what to return.
                This should be left at `info_trajectory`, the
                other values are for internal usage with subroutines.
            init_model_stats: Initial model stats to use for the run.

        Returns:
            If return_type is "info_trajectory", returns a tuple of
            the info dictionary and the trajectory (list of dictionaries).
        """
        if env.container_obj.id != self.last_container_id:
            self.logger.info(f"Initializing agent settings for container {env.container_obj.id}")
            self.init_environment_vars(env)
            self.last_container_id = env.container_obj.id

        self.setup(setup_args, init_model_stats)
        for hook in self.hooks:
            hook.on_run_start()

        trajectory = []
        info = {}
        traj_log_path = traj_dir / (env.record["instance_id"] + ".traj")
        self.logger.info("Trajectory will be saved to %s", traj_log_path)

        done = False
        while not done:
            for hook in self.hooks:
                hook.on_step_start()
            state = env.communicate(self.state_command) if self.state_command else None
            thought, action, output = self.forward(observation, env.get_available_actions(), state)
            for hook in self.hooks:
                hook.on_actions_generated(thought=thought, action=action, output=output)

            observations = list()
            self.logger.info(f"🤖 THOUGHT: {thought}")
            self.logger.info(f"🤖 ACTION: {action}")
            if action is not None:
                run_action = self._guard_multiline_input(action)
                for sub_action in self.split_actions(run_action):
                    if sub_action["agent"] == self.name or sub_action["cmd_name"] == self.config.submit_command:
                        for hook in self.hooks:
                            hook.on_sub_action_started(sub_action=sub_action)
                        obs, _, done, info = env.step(sub_action["action"])
                        for hook in self.hooks:
                            hook.on_sub_action_executed(obs=obs, done=done)
                        observations.append(obs)
                        if sub_action["cmd_name"] == self.config.submit_command:
                            done = True
                        if done:
                            break
                    else:
                        agent_name = sub_action["agent"]
                        sub_agent_output = self.call_subroutine(agent_name, sub_action, env)
                        observations.append(sub_agent_output)

                observation = "\n".join([obs for obs in observations if obs is not None])
            else:
                # sometimes agent select no tools and content is empty. and finish_reason is stop.
                observation = "You must choose one of the functions. Please try again."

            trajectory_step = TrajectoryStep(
                {
                    "action": action,
                    "observation": observation,
                    "response": output,
                    "state": state,
                    "thought": thought,
                },
            )
            trajectory.append(trajectory_step)
            model_stats: APIStats = self.model.stats
            info["model_stats"] = model_stats.to_dict()
            if traj_dir:
                self.save_trajectory(trajectory, traj_log_path, env_name=env.name, info=info)
            for hook in self.hooks:
                hook.on_step_done(trajectory_step=trajectory_step, model_stats=model_stats)

        for hook in self.hooks:
            hook.on_run_done()
        
        self.logger.info("Trajectory saved to %s", traj_log_path)

        if return_type == "info":
            return info
        if return_type == "info_trajectory":
            return info, trajectory
        return trajectory[-1][return_type]


    def forward(self, observation: str, available_actions: list[str], state: str) -> tuple[str, str, str]:
        """Forwards the model

        Args:
            observation: Observation
            available_actions: Currently not used
            state:

        Returns:
            thought: model reasoning
            action: action that the model proposes
            output: raw model output (not output of the action)
        """

        tools = []
        for cmd in self.config._commands:
            tools.append(self.command2fncall(cmd))

        if state is None:
            state_vars = {}
        else:
            state_vars = json.loads(state)
        #print("!! instance_args="+str(self.instance_args))
        #print("!! system_args="+str(self.system_args))
        #print("!! state_vars="+str(state_vars))

        if len(self.history) == 0:
            system_message_content = self.config.system_template.format(
                **self.instance_args,
                **self.system_args,
                **state_vars
            )
            self._append_history(
                {
                    "role" : "system",
                    "content": system_message_content
                }
            )

            instance_message_content = self.config.instance_template.format(
                **self.instance_args,
                **self.system_args,
                **state_vars
            )
            self._append_history(
                {
                    "role" : "user",
                    "content": instance_message_content
                }
            )


        if observation is not None:
            content = "{observation}\n(Open file: {open_file})\n(Current directory: {working_dir})".format(
                observation=observation,
                **state_vars
            )
            self.logger.info(f"🤖 OBSERVATION: {content}")
            # find tool call id from history
            if "tool_calls" in self.history[-1]:
                tool_calls = self.history[-1]["tool_calls"]
                self._append_history(
                    {
                        "role" : "tool",
                        "tool_call_id": tool_calls[0]["id"],
                        "content": content,                    
                    }
                )
            else:
                # Tool call not found from history but observation value is provided
                # e.g. when agent select no tools but finish_reason is still not stop
                # in this case, we add observation as a user message
                self._append_history(
                    {
                        "role" : "user",
                        "content": content
                    }
                )

        for hook in self.hooks:
            hook.on_model_query(query=self.local_history, agent=self.name)

        # call model
        #self.logger.info(f"🤖 MODEL INPUT\n{self.history}\n[tools]\n{tools}")
        choice = self.model.query(self.history, tools)
        m = choice["message"]
        self._append_history(m)
        finish_reason = choice["finish_reason"]
        if finish_reason == "stop":
            thought = m["content"]
            action = None
            output = json.dumps(choice)
            return thought, action, output
        elif finish_reason == "tool_calls":
            thought = m["content"]
            action = m["tool_calls"][0]["function"]["name"]
            arguments = json.loads(m["tool_calls"][0]["function"]["arguments"])
            for k, v in arguments.items():
                action = action + " " + str(v)            
            output = json.dumps(choice)
            return thought, action, output
        else:
            raise ValueError("Unexpected finish reason: " + finish_reason)

    def command2fncall(self, cmd: Command) -> dict:
        properties = {}
        required = []

        if cmd.arguments:
            for k, v in cmd.arguments.items():
                properties[k] = {
                    "type": v["type"],
                    "description": v["description"],
                }
                if v["required"]:
                    required.append(k)

        fn = {
            "type": "function",
            "function": {
                "name": cmd.name,
                "description": cmd.docstring,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        
        return fn
    
