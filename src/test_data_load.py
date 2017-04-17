import dataprocessor2 as dp2
import time

a = time.time()
gn_arr = dp2.unwind_games('../ficsgamesdb_2015_CvC_nomovetimes_1443974.pgn')
a = time.time() - a
print("Took %s seconds to load all games" % a)
