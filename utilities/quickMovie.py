from openmmlib import pymol_show
import sys, os

if len(sys.argv) != 5:
    print("---------wrong arguments---------")
    print("usage python movie.py trajectoryPath moviePath start end;")
    print("example: python movie.py myTrajectory movieFolder 1 30")

moviepath = sys.argv[2]

if os.path.exists(moviepath):
    if os.listdir(moviepath):
        print("Movie path should not exist, or be empty")
        raise ValueError("Movie path contains files; please clear it before restarting")
else:
    os.mkdir(moviepath)

trajectoryPath = sys.argv[1]

files = [
    os.path.join(trajectoryPath, "block{0}.dat".format(i))
    for i in range(int(sys.argv[3]), int(sys.argv[4]))
]

pymol_show.makeMoviePymol(
    files, moviepath, fps=20,
)
