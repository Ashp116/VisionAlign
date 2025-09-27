import argparse, os, math, random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from align.utils import add_nuisance, warp_similarity

def load_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def augment_pair(master_rgb, max_shift, max_rot_deg, max_log_scale):
    H, W, _ = master_rgb.shape
    # sample forward transform master->current
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    theta = np.deg2rad(np.random.uniform(-max_rot_deg, max_rot_deg))
    rho = np.random.uniform(-max_log_scale, max_log_scale)

    # synthesize current
    current_rgb = warp_similarity(master_rgb, dx, dy, theta, rho)
    current_rgb = add_nuisance(cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR))
    current_rgb = cv2.cvtColor(current_rgb, cv2.COLOR_BGR2RGB)

    # label is the inverse (current->master), which is simply (-dx, -dy, -theta, -rho) for similarity
    y = np.array([-dx, -dy, -theta, -rho], dtype=np.float32)
    return master_rgb, current_rgb, y

def data_gen(master_path, img_size, batch, max_shift, max_rot_deg, max_log_scale):
    master_rgb = load_image(master_path, img_size)
    while True:
        ms = np.zeros((batch, img_size, img_size, 3), dtype=np.float32)
        cs = np.zeros_like(ms)
        ys = np.zeros((batch, 4), dtype=np.float32)
        for i in range(batch):
            m, c, y = augment_pair(master_rgb, max_shift, max_rot_deg, max_log_scale)
            ms[i] = m / 255.0
            cs[i] = c / 255.0
            ys[i] = y
        yield [ms, cs], ys

def build_model(img_size):
    IMG = (img_size, img_size, 3)
    # MobileNetV2 backbone
    base = tf.keras.applications.MobileNetV2(input_shape=IMG, include_top=False, weights='imagenet')
    x = layers.GlobalAveragePooling2D()(base.output)
    enc = models.Model(base.input, x, name='encoder')

    inp_m = layers.Input(shape=IMG, name='master')
    inp_c = layers.Input(shape=IMG, name='current')

    pre = tf.keras.applications.mobilenet_v2.preprocess_input
    fm = enc(pre(inp_m))
    fc = enc(pre(inp_c))

    h = layers.Concatenate()([fm, fc])
    h = layers.Dense(256, activation='relu')(h)
    h = layers.Dense(128, activation='relu')(h)
    out = layers.Dense(4, name='transform')(h)  # dx, dy, theta(rad), rho(log-scale)

    model = models.Model([inp_m, inp_c], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.Huber())
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master', required=True, help='Path to master/reference image')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--steps', type=int, default=8000, help='Training steps (batches)')
    ap.add_argument('--val_steps', type=int, default=500)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--max_shift', type=float, default=40.0)
    ap.add_argument('--max_rot_deg', type=float, default=10.0)
    ap.add_argument('--max_log_scale', type=float, default=0.15)
    ap.add_argument('--out', default='models/siamese_v1.keras')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = build_model(args.img_size)
    model.summary()

    train_ds = tf.data.Dataset.from_generator(
        lambda: data_gen(args.master, args.img_size, args.batch, args.max_shift, args.max_rot_deg, args.max_log_scale),
        output_signature=(
            (tf.TensorSpec(shape=(None, args.img_size, args.img_size, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(None, args.img_size, args.img_size, 3), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: data_gen(args.master, args.img_size, args.batch, args.max_shift*0.7, args.max_rot_deg*0.7, args.max_log_scale*0.7),
        output_signature=(
            (tf.TensorSpec(shape=(None, args.img_size, args.img_size, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(None, args.img_size, args.img_size, 3), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    ckpt = tf.keras.callbacks.ModelCheckpoint(args.out, monitor='val_loss', save_best_only=True)
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(train_ds, steps_per_epoch=args.steps,
                        validation_data=val_ds, validation_steps=args.val_steps,
                        epochs=1, callbacks=[ckpt, early])

    model.save(args.out)
    print(f"Saved model to {args.out}")

if __name__ == '__main__':
    main()
