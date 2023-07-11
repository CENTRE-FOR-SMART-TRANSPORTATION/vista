

default: kill
# @kill $(jobs -p)
kill:
	@kill $(jobs -p)