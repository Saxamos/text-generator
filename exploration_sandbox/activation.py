# Récupération des activations (valeurs) d'une couche donnée
def predict_activations(complete_model, predicted_paragraph, tar_layer):
    activations = []
    # Création du modèle tronqué
    partial_model = Sequential()
    for l in range(tar_layer):
        partial_model.add(complete_model.layers[l])
    partial_model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Récupération des outputs de la couche à chaque caractère prédit
    for i in range(len(predicted_paragraph) - SEQ_LEN):
        sentence = predicted_paragraph[i:i + SEQ_LEN]
        act = predict_single_input(partial_model, sentence)
        activations.append(act)
    activations = np.transpose(activations)
    fs_l_acts = [[0] * SEQ_LEN + list(a) for a in activations]
    return np.array(fs_l_acts)


# Récupération de toutes les activations du réseau
def all_activations(complete_model, predicted_paragraph):
    net_acts = {}
    for layer in range(1, 1 + len(MODEL.layers)):
        net_acts[layer] = predict_activations(MODEL, predicted_paragraph, layer)
    return net_acts
