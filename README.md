# E94116041_numerical_hw4
數值方法 作業4
第1題程式碼
  import numpy as np
  定義被積函數
  def f(x):
      return np.exp(x) * np.sin(4 * x)
  給定參數
  a = 1.0  # 積分下限
  b = 2.0  # 積分上限
  h = 0.1  # 步長
  n = int((b - a) / h)  # 子區間數量，n = 10
  積分點
  x = np.linspace(a, b, n + 1)  # x = [1.0, 1.1, ..., 2.0]
  f_values = f(x)  # 計算 f(x) 在每個點的值
  a. 複合梯形法則
  def composite_trapezoidal_rule(f_values, h, n):
      result = (h / 2) * (f_values[0] + 2 * np.sum(f_values[1:n]) + f_values[n])
      return result
  b. 複合辛普森法則
  def composite_simpson_rule(f_values, h, n):
      if n % 2 != 0:
          raise ValueError("n must be even for Simpson's rule")
      result = (h / 3) * (f_values[0] + 4 * np.sum(f_values[1:n:2]) + 2 * np.sum(f_values[2:n-1:2]) + f_values[n])
      return result
  c. 複合中點法則
  def composite_midpoint_rule(f, x, h, n):
      midpoints = (x[:-1] + x[1:]) / 2  # 中點：(x_i + x_{i+1})/2
      result = h * np.sum(f(midpoints))
      return result
  計算結果
  trapezoidal_result = composite_trapezoidal_rule(f_values, h, n)
  simpson_result = composite_simpson_rule(f_values, h, n)
  midpoint_result = composite_midpoint_rule(f, x, h, n)
  輸出結果
  print(f"a. Composite Trapezoidal Rule: {trapezoidal_result:.6f}")
  print(f"b. Composite Simpson's Rule: {simpson_result:.6f}")
  print(f"c. Composite Midpoint Rule: {midpoint_result:.6f}")
  
  為了驗證，計算真實值（可通過數值積分或解析解）
  from scipy.integrate import quad
  true_value, _ = quad(f, a, b)
  print(f"\nTrue value (for reference): {true_value:.6f}")
  
第2題程式碼

  import numpy as np
  from scipy.special import roots_legendre
  from scipy.integrate import quad
  定義被積函數
  def f(x):
      return x**2 * np.log(x)
  高斯求積法
  def gaussian_quadrature(f, a, b, n):
      # 獲取高斯-勒讓德節點和權重
      t, w = roots_legendre(n)
      # 變量變換：t -> x
      x = (b - a) / 2 * t + (a + b) / 2
      # 積分公式
      result = (b - a) / 2 * np.sum(w * f(x))
      return result
  積分區間
  a = 1.0
  b = 1.5
  當n = 3
  result_n3 = gaussian_quadrature(f, a, b, 3)
  print(f"Gaussian Quadrature (n=3): {result_n3:.6f}")
  當n = 4
  result_n4 = gaussian_quadrature(f, a, b, 4)
  print(f"Gaussian Quadrature (n=4): {result_n4:.6f}")
  真實值
  true_value, _ = quad(f, a, b)
  print(f"True value: {true_value:.6f}")
  計算相對誤差
  relative_error_n3 = abs(result_n3 - true_value) / abs(true_value)
  relative_error_n4 = abs(result_n4 - true_value) / abs(true_value)
  轉換為百分比
  relative_error_n3_percent = relative_error_n3 * 100
  relative_error_n4_percent = relative_error_n4 * 100
  
  print(f"Relative error (n=3, %): {relative_error_n3_percent:.6f}%")
  print(f"Relative error (n=4, %): {relative_error_n4_percent:.6f}%")

第3題程式碼

  import numpy as np
  from scipy.integrate import dblquad
  定義被積函數
  def f(x, y):
      return 2 * y * np.sin(x) + np.cos(x)**2
  複合辛普森法則（二重積分）
  def composite_simpson_2d(f, a, b, c, d, n, m):
      hx = (b - a) / n
      x = np.linspace(a, b, n + 1)
      I = np.zeros(n + 1)
      for i in range(n + 1):
          xi = x[i]
          ci = c(xi)
          di = d(xi)
          hy = (di - ci) / m
          y = np.linspace(ci, di, m + 1)
          fy = f(xi, y)
          I[i] = (hy / 3) * (fy[0] + 4 * np.sum(fy[1:m:2]) + 2 * np.sum(fy[2:m-1:2]) + fy[m])
      result = (hx / 3) * (I[0] + 4 * np.sum(I[1:n:2]) + 2 * np.sum(I[2:n-1:2]) + I[n])
      return result
  高斯求積法（二重積分）
  def gaussian_quadrature_2d(f, a, b, c, d, n, m):
      from scipy.special import roots_legendre
      tx, wx = roots_legendre(n)
      x = (b - a) / 2 * tx + (a + b) / 2
      I = np.zeros(n)
      for i in range(n):
          xi = x[i]
          ci = c(xi)
          di = d(xi)
          ty, wy = roots_legendre(m)
          y = (di - ci) / 2 * ty + (ci + di) / 2
          fy = f(xi, y)
          I[i] = (di - ci) / 2 * np.sum(wy * fy)
      result = (b - a) / 2 * np.sum(wx * I)
      return result
  積分區間
  a = 0
  b = np.pi / 4
  c = lambda x: np.sin(x)
  d = lambda x: np.cos(x)
  a. 複合辛普森法則 (n=4, m=4)
  simpson_result = composite_simpson_2d(f, a, b, c, d, 4, 4)
  print(f"a. Composite Simpson's Rule (n=4, m=4): {simpson_result:.6f}")
  b. 高斯求積法 (n=3, m=3)
  gauss_result = gaussian_quadrature_2d(f, a, b, c, d, 3, 3)
  print(f"b. Gaussian Quadrature (n=3, m=3): {gauss_result:.6f}")
  c. 真實值
  true_value, _ = dblquad(lambda y, x: f(x, y), a, b, c, d)
  print(f"c. True value: {true_value:.6f}")
  計算相對誤差並轉換為百分比
  simpson_relative_error = abs(simpson_result - true_value) / abs(true_value)
  gauss_relative_error = abs(gauss_result - true_value) / abs(true_value)
  simpson_relative_error_percent = simpson_relative_error * 100
  gauss_relative_error_percent = gauss_relative_error * 100
  
  print(f"\nRelative error (Simpson, %): {simpson_relative_error_percent:.6f}%")
  print(f"Relative error (Gaussian, %): {gauss_relative_error_percent:.6f}%")

第4題程式碼

  import numpy as np
  複合辛普森法則
  def composite_simpson(f, a, b, n):
      if n % 2 != 0:
          raise ValueError("n must be even for Simpson's rule")
      h = (b - a) / n
      x = np.linspace(a, b, n + 1)
      fx = f(x)
      result = (h / 3) * (fx[0] + 4 * np.sum(fx[1:n:2]) + 2 * np.sum(fx[2:n-1:2]) + fx[n])
      return result
  積分 a: \int_0^1 x^{-1/4} \sin x \, dx
  變量變換 t = x^{-1}，積分變為 \int_1^\infty t^{-7/4} \sin(t^{-1}) \, dt
  截斷到 [1, 100]
  def g1(t):
      return t**(-7/4) * np.sin(t**(-1))
  result_a = composite_simpson(g1, 1, 100, 4)
  print(f"a. Integral (0 to 1) x^(-1/4) sin x dx: {result_a:.6f}")
  積分 b: \int_1^\infty x^{-4} \sin x \, dx
  變量變換 t = x^{-1}，積分變為 \int_0^1 t^2 \sin(t^{-1}) \, dt
  截斷到 [0.01, 1]
  def g2(t):
      return t**2 * np.sin(t**(-1))
  result_b = composite_simpson(g2, 0.01, 1, 4)
  print(f"b. Integral (1 to inf) x^(-4) sin x dx: {result_b:.6f}")
