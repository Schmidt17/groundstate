import time


class Timer:
	_instance_counter = 0
	"""A slim context manager for timing code blocks"""
	def __init__(self, name=None):
		Timer._instance_counter += 1

		self.name = name
		if self.name is None:
			self.name = f'Block {Timer._instance_counter}'

	def __enter__(self):
		self.t_wall_start = time.perf_counter()
		self.t_cpu_start = time.process_time()

	def __exit__(self, type, value, traceback):
		self.t_wall_end = time.perf_counter()
		self.wall_dur = self.t_wall_end - self.t_wall_start

		self.t_cpu_end = time.process_time()
		self.cpu_dur = self.t_cpu_end - self.t_cpu_start

		print(f"{self.name} took {self.wall_dur:.2e} s ({self.cpu_dur:.2e} s CPU time).")