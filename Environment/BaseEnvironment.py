from abc import ABC, abstractmethod

class Environment(ABC):
	@abstractmethod
	def get_env_path(self):
		pass

	@abstractmethod
	def get_home_team_size(self):
		pass

	@abstractmethod
	def get_away_team_size(self):
		pass