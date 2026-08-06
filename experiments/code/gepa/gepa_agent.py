from appworld import AppWorld
from appworld.common.constants import DEFAULT_EXPERIMENT_NAME
from appworld_experiments.code.ace.evaluation_agent import Agent, ExecutionIO

from appworld.evaluator import evaluate_task

class GEPAAgent(Agent):
    def __init__(
        self,
        generator_model_config: dict,
        appworld_config: dict | None = None,
        logger_config: dict | None = None,
        max_steps: int = 10,
        max_cost_overall: float = 3000,
        max_cost_per_task: float = 10,
        log_lm_calls: bool = False,
    ):
        super().__init__(
            generator_model_config=generator_model_config,
            appworld_config=appworld_config,
            logger_config=logger_config,
            max_steps=max_steps,
            max_cost_overall=max_cost_overall,
            max_cost_per_task=max_cost_per_task,
            log_lm_calls=log_lm_calls
        )

    def solve_task(self, task_id: str, experiment_name: str | None = None):
        experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        self.cost_tracker.reset(task_id)

        self.initial_code_idx = None
        self.previous_code_idx = None
        self.previous_error_idx = None
        reflections = []
        test_tracker = None
        
        with AppWorld(
            task_id=task_id, experiment_name=experiment_name, **self.appworld_config
        ) as world:
            execution_outputs: list[ExecutionIO] = []
            self.initialize(world)

            print("---Max steps---: ", self.max_steps)
            for _ in range(self.max_steps):
                self.step_number += 1
                execution_inputs, cost, reflection = self.next_execution_inputs_and_cost(execution_outputs, "")
                if reflection:
                    reflections.append(reflection)

                if len(execution_inputs) != 0:
                    execution_outputs = [
                        ExecutionIO(
                            content=world.execute(execution_input.content),
                            metadata=execution_input.metadata,
                        )
                        for execution_input in execution_inputs
                    ]
                    
                    # Show execution results to user via logger
                    for i, output in enumerate(execution_outputs):
                        if output.content.strip():  # only show non-empty outputs
                            self.logger.show_message(
                                role="environment", 
                                message=output.content, 
                                step_number=self.step_number
                            )
                    
                    self.cost_tracker.add(task_id, cost)
                    self.log_cost()

                if world.task_completed() or self.cost_tracker.exceeded():
                    test_tracker, _ = evaluate_task(task_id, experiment_name)
                    break

            if test_tracker is None:
                test_tracker = [execution_output.content for execution_output in execution_outputs]
                        
        self.logger.complete_task()
        return test_tracker