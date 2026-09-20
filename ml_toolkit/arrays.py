def array_sum_and_cnt(a):
  summ = 0
  cnt = 0
  for i in a:
    # Для надежности в ML лучше заменить type(i) == list на isinstance
    if isinstance(i, list):
      sum1, cnt1 = array_sum_and_cnt(i)
      cnt += cnt1
      summ += sum1
    elif isinstance(i, (int, float)):
      summ += i
      cnt += 1
  return summ, cnt

def array_sum(a):
  summ, cnt = array_sum_and_cnt(a)
  return summ


def array_mean(a):
  summ, cnt = array_sum_and_cnt(a)
  if cnt == 0:  # Защита от деления на ноль
    return 0.0
  return summ / cnt


def array_min(a):
  # Используем настоящую математическую бесконечность
  current_min = float('inf')
  for i in a:
    if isinstance(i, list):
      min1 = array_min(i)
      if min1 <= current_min:
        current_min = min1
    elif isinstance(i, (int, float)) and i <= current_min:
      current_min = i
  return current_min


def array_max(a):
  # Используем настоящую отрицательную бесконечность
  current_max = float('-inf')
  for i in a:
    if isinstance(i, list):
      max1 = array_max(i)
      if max1 >= current_max:
        current_max = max1
    elif isinstance(i, (int, float)) and i >= current_max:
      current_max = i
  return current_max


def array_shape(a):
  if not isinstance(a, list):
    return ()
  if len(a) == 0:
    return (0,)
  # Защита: если первый подэлемент — пустой список, не идем вглубь через a[0]
  if isinstance(a[0], list) and len(a[0]) == 0:
    return (len(a), 0)

  return (len(a),) + array_shape(a[0])