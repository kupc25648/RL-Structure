'''
=====================================================================
Master file for QLEARNING for training frame structure
This is the master file for using Q-Learning alogorithm for sturcutral optimization problems.
Adjustable parameter are under '研究室'

フレーム構造をトレーニングするためのQLEARNINGのマスターファイル
これは、構造最適化問題にQ-Learningアルゴリズムを使用するためのマスターファイルです。
調整可能なパラメータは「研究室」の下にあります
=====================================================================
'''

#====================================================================
#Import Part
#インポート部
#====================================================================
from FEM_frame import *
from frame_GEN import *
from frame_RL import *
from frame_ENV_general import *
from keras import backend as K
import ast

import pandas as pd
import datetime
np.random.seed(0) # deterministic random 決定論的ランダム
#====================================================================
#Parameter Part
#パラメータ部
#====================================================================
#----------------------------------
# frame_ENV parameters 研究室
# frame_ENV パラメータ
#----------------------------------
game_end_step = 10 # Game end step / ゲーム終了ステップ
game_alpha = 0.1 # Game alpha value for adjusting structure /構造を調整するためのゲームのアルファ値
game_max_y_val = 1 # Maximum y value(vertical) / yの最大値（垂直）(m)
game_num_agents= 1 # Number of agent(always 1 for Q-Learning) / エージェントの数（Qラーニングの場合は常に1）
game_render = 0 # Render(should be always 0) / レンダリング（常に0である必要があります）
game_tell_action=False # Tell agent's action in each step / ＃各ステップでエージェントのアクションを伝える
objective = 'strain-energy' # objective of this game / ＃このゲームの目的
game_type = 4 #game_type difines number of input of neural network agent [game_type:num_ob , 1:7 , 2:4 , 3:9 , 4:13] / game_typeはニューラルネットワークエージェントの入力数を定義します [game_type:num_ob , 1:7 , 2:4 , 3:9 , 4:13]
#----------------------------------
# frame_GEN parameters 研究室
# frame_GEN パラメータ
#----------------------------------
num_x = 5 # Nodes in x direction(horizontal) / x方向のノード（水平）
num_z = 5 # Nodes in z direction(horizontal) / z方向のノード（水平）
span  = 1 # structural span length(m.) / 構造スパン長さ（m。）
diameter = 0.1 # structural element diameter(m.) / 構造要素の直径（m。）
loadx = 0 # load in x direction (N) / x方向の荷重（N）
loady = -1000 # load in y direction (vertical)(N) / Y方向の荷重（垂直）（N）
loadz = 0 # load in z direction (N) / z方向の荷重（N）
Young = 10000000000 # structural elements' young modulus(N/m2) / 構造要素のヤング率（N / m2）
c1,c2,c3,c4,c5,c6,c7 = 0,0,0,0,0,0,0 # model generate function / ＃モデル生成関数
forgame = 1000 # forgame defines what kind of game/ forgameは、どのようなゲームを定義します
#----------------------------------
# frame_RL parameters (Q_L) 研究室
# frame_RL パラメータ (Q-Learning)
#----------------------------------
lr = 0.001 # neural network agent learning rate / ＃ニューラルネットワークエージェントの学習率
ep = 0.9 # initial epsilon value / 初期イプシロン値
epd = 0.99995 # epsilon decay value / イプシロン減衰値
gamma = 0.9 # reward discount factor / 報酬割引係数
nn = [64,64,64] # how neural network is built(neuron in each layer) / ニューラルネットワークの構築方法（各層のニューロン）
max_mem = 1000000 # maximum length of replay buffer / 再生バッファの最大長
num_ob = 23 #[num_ob  13] nueral network input(state) (related to game_type) / ニューラルネットワーク入力(状態)（game_typeに関連）
num_action = 6 #[class Game:num_ob , class Game0:6 , class Game1:6 , class Game2:3] neural network output(action) / ニューラルネットワーク出力（行動）
train_period = 1 #(0/1) 0=練習なし　, 1=練習あり
#----------------------------------
# How many game to train 研究室
# トレーニングするゲームの数
#----------------------------------
num_episodes = 1
#----------------------------------
# Load reinforcement learning model 研究室
# 強化学習モデルの読み込み
#----------------------------------
base_num=0
#----------------------------------
# How many game to save neural network 研究室
# ニューラルネットワークを保存するゲームの数
#----------------------------------
save_iter = 1

#====================================================================
# Function Part
# 機能部
#====================================================================

#----------------------------------
# Plot function: Function to plot a graph of objective-step in each game
# プロット関数：各ゲームの目的ステップのグラフをプロットする関数
#----------------------------------
log_energy_store = []
game_reward = [0]
now = str(datetime.datetime.now().strftime("_%Y-%m-%d_%H_%M_%S"))
def plotforgame(objective):
    objective = objective
    name_of_file = 'Game_{}_Initial_{}_Final_{}_Reward_{}.png'.format(
        counter+1,
        log_energy_store[0],
        log_energy_store[-1],
        game_reward[0][0])
    save_path = '{}-Step'.format(objective)+now+'/'
    if not os.path.exists('{}-Step'.format(objective)+now):
        os.makedirs('{}-Step'.format(objective)+now)
    name = os.path.join(save_path, name_of_file)
    plt.ylabel('{}'.format(objective))
    plt.xlabel("step")
    plt.plot(log_energy_store)
    plt.savefig(name)
    plt.close("all")

#----------------------------------
# Reinforcement Learning function: Function to run reinforcement learning loop for 1 game
# 強化学習機能：1ゲームの強化学習ループを実行する機能
#----------------------------------
def run(game,objective,train_period=1):
    objective = objective
    env1_test.reset()
    intcounter = 0

    for_out  = []
    for_outz = []
    for_out_ob = []
    while env1_test.over != 1:
        sub_state = []
        sub_action =[]
        sub_nextstate = []
        sub_reward = []
        sub_done =[]

        env1_test.game._game_gen_state_condi()
        if intcounter == 0:
            log_energy_store.append(env1_test.game.int_strain_e[0])
            intcounter = 1
            #-----------------------------
            #save file as .txt for initial model / 初期モデルのファイルを.txtとして保存
            name_of_file = 'Game{}_Step{}.txt'.format(counter+1,0)
            save_path = 'Q_Model_data_txt_Game{}/'.format(counter+1)
            if not os.path.exists('Q_Model_data_txt_Game{}/'.format(counter+1)):
                os.makedirs('Q_Model_data_txt_Game{}/'.format(counter+1))
            name = os.path.join(save_path, name_of_file)
            env1_test.game.gen_model.savetxt(name)

            #----------------------------------
            # for .txt
            sub_outz = []
            for num in range(len(env1_test.game.model.nodes)):
                sub_outz.append(env1_test.game.model.nodes[num].coord[1])
            for_out.append('{} {} {} {}'.format(0,env1_test.game.int_strain_e[0],env1_test.game.reward_counter[0],env1_test.game.gen_model.n_u_name_div[env1_test.game.doing[0][0]][env1_test.game.doing[0][1]].name))
            for_outz.append(sub_outz)
            #----------------------------------

        else:
            pass
        #----------------------------------
        # Reinforcement Learning Loop for 1 step in a game (研究室)
        # ゲームの1ステップの強化学習ループ
        s = env1_test.game._game_get_1_state(env1_test.game.doing[0])
        sub_state.append(s)
        s.reshape((reinforcement_learning.num_state,1))
        action = reinforcement_learning.act(s)
        sub_action.append(action)
        sub_nextstate.append(env1_test.game._game_get_next_state(env1_test.game.doing[0],action))
        reinforcement_learning.remember(
            sub_state[-1],sub_action[-1],0,sub_nextstate[-1],0)
        env1_test.game._gen_gen_reward_condition()
        r,d = env1_test.game._game_get_reward(0)
        sub_reward.append(r)
        sub_done.append(d)
        reinforcement_learning.temprp[-1][2] = sub_reward[-1]
        reinforcement_learning.temprp[-1][4] = sub_done[-1]
        env1_test.game.reward_counter[0] += sub_reward[-1]

        #----------------------------------
        # for .txt
        sub_outz = []
        for num in range(len(env1_test.game.model.nodes)):
            sub_outz.append(env1_test.game.model.nodes[num].coord[1])
        for_out.append('{} {} {} {}'.format(env1_test.game.game_step,env1_test.game.next_strain_e[0],round(env1_test.game.reward_counter[0],2),env1_test.game.gen_model.n_u_name_div[env1_test.game.doing[0][0]][env1_test.game.doing[0][1]].name))
        for_outz.append(sub_outz)

        #----------------------------------
        # train the neural network agent / ニューラルネットワークエージェントの 練習
        if train_period==1:
            reinforcement_learning.train()
        else:
            pass
        # reinforcement_learning.update() # use for Double-DQN, actor critic and DDPG / Double-DQN、俳優評論家、DDPGに使用
        #----------------------------------
        # Add data for the graph / グラフのデータを追加する
        log_energy_store.append(env1_test.game.next_strain_e[0])
        game_reward[0] = env1_test.game.reward_counter
        #----------------------------------
        # Print out result on the console / 結果をコンソールに出力する
        print('Step {} {} int {} {} now {} Reward {} Epsilon {}'.format(
            env1_test.game.game_step,
            objective,
            env1_test.game.int_strain_e[0],
            objective,
            env1_test.game.next_strain_e[0],
            env1_test.game.reward_counter[0],
            reinforcement_learning.ep))
        #----------------------------------
        # Render (should not be use else it will be slow) / レンダリング（使用しないでください。遅くなります）
        if env1_test.game.render == 1:
            name_of_file = 'Step {} {} int{} {} now{} %{} Reward {}.png'.format(
                env1_test.game.game_step,
                objective,
                env1_test.game.int_strain_e[0],
                objective,
                env1_test.game.next_strain_e[0],
                ((env1_test.game.next_strain_e[0] - env1_test.game.int_strain_e[0])/env1_test.game.int_strain_e[0])*100,
                env1_test.game.reward_counter[0])
            save_path = '{}-step/'.format(objective)
            if not os.path.exists('{}-step/'.format(objective)):
                os.makedirs('{}-step/'.format(objective))
            name = os.path.join(save_path, name_of_file)
            env1_test.game.gen_model.render(name,env1_test.game.game_step,env1_test.game.int_strain_e[0],env1_test.game.next_strain_e[0])
        #----------------------------------
        # Game move to nextstep and check if the end condition is met / ゲームは次のステップに移動し、終了条件が満たされているかどうかを確認します
        env1_test.game.step()
        env1_test.check_over()
    # ------------------------------
    # Write and save output model file
    # 出力モデルファイルの書き込みと保存
    # ------------------------------
    gamename_out1 = 'Q_Model_data_txt_Game{}/out{}.txt'.format(counter+1,game)
    gamename_out2 = 'Q_Model_data_txt_Game{}/outz{}.txt'.format(counter+1,game)
    # ------------------
    # out1
    # ------------------
    new_file1 = open(gamename_out1, "w+")
    for i in range(len(for_out)):
        new_file1.write(" {}\r\n".format(for_out[i]))
    new_file1.close()
    # ------------------
    # out2
    # ------------------
    new_file2 = open(gamename_out2, "w+")
    for i in range(len(for_outz)):
        for j in range(len(for_outz[i])):
            new_file2.write("{} ".format(for_outz[i][j]))
        new_file2.write("\n")
    new_file2.close()

    env1_test.reset()

# ----------------------------------
# Save and Restore Neural Network Agent
# ニューラルネットワークエージェントの保存と復元
# ----------------------------------
if base_num != 0:
    base_name_of_Q_Learning_pickle = "Q_Learning_pickle.h5"
    base_pickle_path = '{}pickle_base/'.format(base_num)
    base_Q_Learning_picklename = os.path.join(base_pickle_path, base_name_of_Q_Learning_pickle)
    memory_name = "kioku_no_uta_vocaloid.txt"
    mem = os.path.join(base_pickle_path, memory_name)
# ----------------------------------
# Main program
# メインプログラム
# ----------------------------------
with tf.compat.v1.Session() as sess: # Using tensorflow's calculation / ＃tensorflowの計算を使用する
    reinforcement_learning = Q_Learning(lr,ep,epd,gamma,nn,max_mem,num_ob,num_action,sess) # Make an agent / エージェントを作る
    if base_num!=0: # Load model / モデルをロード
        try:
            #retriveing model
            reinforcement_learning.q_model = load_model(base_Q_Learning_picklename)
            print("Load model success!")
        except:
            print("No model file to restore")
        try:
            #retriveing memeory
            lineList = [line.rstrip('\n') for line in open(mem)]
            for i in range(len(lineList)):
                reinforcement_learning.temprp.append(list(ast.literal_eval(lineList[i])))
                reinforcement_learning.temprp[-1][0] = np.array(reinforcement_learning.temprp[-1][0])
                reinforcement_learning.temprp[-1][1] = np.array(reinforcement_learning.temprp[-1][1])
                reinforcement_learning.temprp[-1][3] = np.array(reinforcement_learning.temprp[-1][3])
            print("Load memory success!")
        except:
            print("No memory file to restore")


    counter = 0 # Counter for game / ゲームのカウンター
    while counter < num_episodes:
        model = gen_model(num_x,num_z,span,diameter,loadx,loady,loadz,Young,c1,c2,c3,c4,c5,c6,c7,forgame,game_max_y_val) # define model / モデルを定義する
        game = Game2(game_end_step,game_alpha,game_max_y_val,model,game_num_agents,game_render,game_tell_action) # choose game /ゲームを選択 (研究室)
        game.set_gametype(game_type) # set game type / ゲームタイプを設定する
        env1_test = ENV(game) # Put the game in ENV / ENVにゲームを置く
        print('Episode{}'.format(counter+1)) # Print the Number of current game in console / 現在のゲームの数をコンソールに出力する
        run(counter+1,objective,train_period) # run reinforcement learning loop / 強化学習ループを実行する
        plotforgame(objective) # plot the graph after game end / ゲーム終了後にグラフをプロットする
        log_energy_store = [] # reset graph data / グラフデータをリセットする
        game_reward = [0] # reset reward / 報酬をリセット
        counter += 1 # game counter += 1

        if counter%save_iter == 0: # save neural network weigth every defined interval / ＃定義された間隔ごとにニューラルネットワークの重みを保存
            name_of_Q_Learning_pickle = "Q_Learning_pickle.h5"
            pickle_path = '{}pickle_base/'.format(counter)
            if not os.path.exists('{}pickle_base/'.format(counter)):
                os.makedirs('{}pickle_base/'.format(counter))
                Q_Learning_picklename = os.path.join(pickle_path, name_of_Q_Learning_pickle)
            Q_Learning_picklename = os.path.join(pickle_path, name_of_Q_Learning_pickle)
            reinforcement_learning.q_model.save(Q_Learning_picklename)
            # ------------------
            # memory
            # ------------------
            memory_name = "kioku_no_uta_vocaloid.txt"
            memory_name_picklename = os.path.join(pickle_path, memory_name)
            memory = open(memory_name_picklename, "w+")
            for i in range(len(reinforcement_learning.temprp)):
                memory.write("{},{},{},{},{} ".format(reinforcement_learning.temprp[i][0].tolist(),reinforcement_learning.temprp[i][1].tolist(),reinforcement_learning.temprp[i][2],reinforcement_learning.temprp[i][3].tolist(),reinforcement_learning.temprp[i][4]))
                memory.write("\n")
            memory.close()
            print("saved")

