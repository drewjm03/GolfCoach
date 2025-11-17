import pickle, numpy as np
p = r"C:\Users\anm5573\Documents\GitHub\GolfCoach\data\AprilBoards2.pickle"
with open(p,"rb") as f:
    data = pickle.load(f)

obj = data["at_board_d"]
print("len:", len(obj))
print("elem0 type:", type(obj[0]))
e = obj[0]
if isinstance(e, dict):
    print("elem0 dict keys:", list(e.keys()))
elif isinstance(e, (list, tuple)):
    print("elem0 tuple/list lens:", len(e))
    if len(e) >= 2:
        print("id?", type(e[0]), " corners type/shape?", type(e[1]), (np.array(e[1]).shape if hasattr(e[1],'__array__') else 'n/a'))
else:
    print("elem0 attrs:", [a for a in dir(e) if not a.startswith('_')][:20])