
// fn predict(self, frame) {
//     // Preprocess
//     input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
//     input_img = cv2.resize(input_img, (CORRECTOR_SIZE, CORRECTOR_SIZE)) / 255.
//     input_img = np.reshape(input_img, [1, CORRECTOR_SIZE, CORRECTOR_SIZE, 3])

//     x, y, w, h = self.sess.run([self.x, self.y, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

//     img_h, img_w, _ = frame.shape

//     x = int(x*img_w)
//     w = int(w*img_w)

//     y = int(y*img_h)
//     h = int(h*img_h)

//     return x, y, w, h
// }
