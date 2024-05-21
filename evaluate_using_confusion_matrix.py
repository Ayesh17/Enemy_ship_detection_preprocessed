import numpy as np

def calculate_metrics(tp, tn, fp, fn):
  """Calculates recall, precision, specificity, and F1 score from a confusion matrix.

  Args:
    confusion_matrix: A 3x3 confusion matrix.

  Returns:
    A tuple containing the recall, precision, specificity, and F1 score.
  """


  # tp = confusion_matrix[1, 1]
  # tn = confusion_matrix[0, 0]
  # fp = confusion_matrix[0, 1]
  # fn = confusion_matrix[1, 0]

  accuracy = (tp + tn) / (tp + fp + tn + fn)
  recall = tp / (tp + fn)
  precision = tp / (tp + fp)
  specificity = tn / (tn + fp)
  f1_score = 2 * (precision * recall) / (precision + recall)

  return accuracy, precision, recall, f1_score


def matrix_conversion(confusion_matrix, pos):
  if pos == 0:
    tp = confusion_matrix[0, 0]
    tn = confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[2, 1] + confusion_matrix[2, 2]
    fp = confusion_matrix[1, 0] + confusion_matrix[2, 0]
    fn = confusion_matrix[0, 1] + confusion_matrix[0, 2]

  elif pos == 1:
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0] + confusion_matrix[0, 2] + confusion_matrix[2, 0] + confusion_matrix[2, 2]
    fp = confusion_matrix[0, 1] + confusion_matrix[2, 1]
    fn = confusion_matrix[1, 0] + confusion_matrix[1, 2]


  else:
    tp = confusion_matrix[2, 2]
    tn = confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 1] + confusion_matrix[1, 2]
    fp = confusion_matrix[0, 2] + confusion_matrix[1, 2]
    fn = confusion_matrix[2, 0] + confusion_matrix[2, 1]

  # if pos == 0 :
  #   tp = confusion_matrix[0, 0]
  #   tn = confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[2, 1] + confusion_matrix[2, 2]
  #   fp = confusion_matrix[0, 1] + confusion_matrix[0, 2]
  #   fn = confusion_matrix[1, 0]  + confusion_matrix[2, 0]
  #
  # elif pos == 1:
  #   tp = confusion_matrix[1, 1]
  #   tn = confusion_matrix[0, 0] + confusion_matrix[0, 2] + confusion_matrix[2, 0] + confusion_matrix[2, 2]
  #   fp = confusion_matrix[1, 0] + confusion_matrix[1, 2]
  #   fn = confusion_matrix[0, 1] + confusion_matrix[2, 1]
  #
  # else:
  #   tp = confusion_matrix[2, 2]
  #   tn = confusion_matrix[0, 0] + confusion_matrix[0, 1] +  confusion_matrix[1, 1] + confusion_matrix[1, 2]
  #   fp = confusion_matrix[2, 0] + confusion_matrix[2, 1]
  #   fn = confusion_matrix[0, 2] + confusion_matrix[1, 0]


  return tp, tn, fp, fn

def weighted_calculation(confusion_matrix):
  btp, btn, bfp, bfn = matrix_conversion(confusion_matrix, 0)
  rtp, rtn, rfp, rfn = matrix_conversion(confusion_matrix, 1)
  bltp, bltn, blfp, blfn = matrix_conversion(confusion_matrix, 2)

  bacc = (btp + btn )/ (btp + btn + bfp + bfn)
  racc = (rtp + rtn )/ (rtp + rtn + rfp + rfn)
  blacc = (bltp + bltn)/ (bltp + bltn + blfp + blfn)

  print(bacc, btp, btn, bfp, bfn)
  print(racc, rtp, rtn, rfp, rfn)
  print(blacc, bltp, bltn, blfp, blfn)

  b_acc, b_precision, b_recall, b_f1_score = calculate_metrics (btp, btn, bfp, bfn)
  r_acc, r_precision, r_recall, r_f1_score = calculate_metrics(rtp, rtn, rfp, rfn)
  bl_acc, bl_precision, bl_recall, bl_f1_score = calculate_metrics(bltp, bltn, blfp, blfn)
  print("b", b_acc, b_precision, b_recall, b_f1_score)
  print("r", r_acc, r_precision, r_recall, r_f1_score)
  print("bl", bl_acc, bl_precision, bl_recall, bl_f1_score)

  b_overall = btp + bfn
  r_overall = rtp + rfn
  bl_overall = bltp + blfn

  accuracy = (bacc * b_overall + r_acc * r_overall + blacc * bl_overall) / (b_overall + r_overall + bl_overall)
  precision = (b_precision * b_overall + r_precision * r_overall + bl_precision * bl_overall) / (b_overall + r_overall + bl_overall)
  recall =  (b_recall * b_overall  +  r_recall * r_overall  +  bl_recall * bl_overall) / (b_overall + r_overall + bl_overall)
  # specificity = (b_specificity * b_overall + r_specificity * r_overall + bl_specificity * bl_overall) / (b_overall + r_overall + bl_overall)
  f1_score = (b_f1_score * b_overall + r_f1_score * r_overall + bl_f1_score * bl_overall) / (b_overall + r_overall + bl_overall)

  return accuracy, precision, recall, f1_score


  # Example usage:

confusion_matrix = np.array([[68, 1, 1],
                              [0, 104,2],
                              [2, 17, 87]])


accuracy, precision, recall, f1_score = weighted_calculation(confusion_matrix)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
# print("Specificity:", specificity)
print("F1 score:", f1_score)
