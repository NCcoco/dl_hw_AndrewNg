import tensorflow as tf

print(tf.__version__)
print(tf.keras.__version__)
# # 自动微分：
# x = tf.Variable(3.0)

# with tf.GradientTape() as tape:
#     y = x**2
    
# dy_dx = tape.gradient(y, x)
# print(dy_dx.numpy())
    
    
# # 当自动微分在函数中时
# def mul(a):
#     return a*3

# a = tf.Variable(5.0)
# with tf.GradientTape() as tape:
#     y = mul(a)
# print(tape.gradient(y, a).numpy())


w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]

c = 0.
for i in range(100):
    # print(w)
    with tf.GradientTape() as tape:
        # @ 表示矩阵乘法
        y = x @ w + b
        loss = tf.reduce_mean(y**2)
    print(loss)
    optimizer = tf.optimizers.Adam(learning_rate=0.00001)
    # print(optimizer.minimize(loss, var_list=[w,b], tape=tape).numpy())




