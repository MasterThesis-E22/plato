import wandb
from plato.config import Config
import logging

class WANDBLogger:
    _projectName = Config().data.datasource
    _groupName = Config().params["experiment_name"]
    _entityName = "master-thesis-22"
    
    def configure(cls, projectName, groupName, entityName) -> None:
        cls._projectName = projectName
        cls._groupName = groupName
        cls._entityName = entityName
        
    
    def __init__(self, runName):
        self._initiated: bool = False
        self.runName = runName
        
        
    def start(self):
        wandb.init(
            project=WANDBLogger._projectName, 
            group=WANDBLogger._groupName, 
            entity=WANDBLogger._entityName
        )
        wandb.run.name = self.runName
        logging.info(
            "[{}] initiated wandb logging for experiment <{}>".format(self.runName, self._groupName)
        )
        self._initiated = True
        
        
    def finish(self) -> None:
        wandb.finish()
        self._initiated = False
    
    
    def log(self,
        data: any,
        step: any = None,
        commit: any = None,
        sync: any = None
    ) -> None:
        if not self._initiated: self.start()
        wandb.log(data, step, commit, sync)