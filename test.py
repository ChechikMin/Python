from classification import *

from torchviz import make_dot

df = pd.read_csv('clean_data.csv', delimiter=',')

model = Classification(9, 1)
model.extract_data(df, aim_par='card')
model.train()
# model.load_state_dict(torch.load('1layer_weights.pt'))
model.predict()
model.show_results()
