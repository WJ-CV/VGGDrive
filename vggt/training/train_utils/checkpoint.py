


def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
    allow_none_state: bool = False,
    detailed_error: bool = True,
    sync_when_finished: bool = False,
):
    """
    Loads a state dict into the given model.

    Args:
        state_dict: A dictionary containing the model's
            state dict, or a subset if strict is False
        model: Model to load the checkpoint weights into
        strict: raise if the state_dict has missing state keys
        ignore_missing_keys: unix pattern of keys to ignore
        checkpoint_kernels: A list of checkpoint processing kernels
        allow_none_state: If true, will not raise error if state_dict is None
    """
    if allow_none_state and state_dict is None:
        logging.info("Interrupted loading of checkpoint: state dict is None.")
        return model

    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        detailed_error=detailed_error,
    )
    logging.warning(f"Loaded {len(state_dict)} keys from checkpoint")

    if sync_when_finished:
        torch.distributed.barrier()
    return model


def check_load_state_dict_errors(
    missing_keys,
    unexpected_keys,
    strict: bool,
    ignore_missing_keys: List[str] = None,
    detailed_error: bool = True,
):
    if ignore_missing_keys is not None and len(ignore_missing_keys) > 0:
        ignored_keys = unix_pattern_to_parameter_names(
            ignore_missing_keys, missing_keys
        )
        missing_keys = [key for key in missing_keys if key not in ignored_keys]

    err = "State key mismatch."
    if unexpected_keys:
        if detailed_error:
            unexpected_keys_str = "\n\t".join(unexpected_keys)
            err += f" Unexpected keys: count: {len(unexpected_keys)}, names:\n\t{unexpected_keys_str}."
        else:
            err += f" Unexpected keys: count: {len(unexpected_keys)}."
    if missing_keys:
        if detailed_error:
            missing_keys_str = "\n\t".join(missing_keys)
            err += f" Missing keys: count: {len(missing_keys)}, names:\n\t{missing_keys_str}."
        else:
            err += f" Missing keys: count: {len(missing_keys)}."
    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)



def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """
    Go through the list of parameter names and select those that match
    any of the provided constraints
    """
    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(
            fnmatch.filter(all_parameter_names, param_name, flags=get_flags())
        )
        assert (
            len(matching_parameters) > 0
        ), f"param_names {param_name} don't match any param in the given names."
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)





def load_multiple_checkpoints_and_apply_kernels(checkpoints: List[Dict]):
    # FIXME: deprecate this method, use merge_checkpoints instead
    state_dicts = []
    for ckpt in checkpoints:
        state_dicts.append(load_checkpoint_and_apply_kernels(**ckpt))

    pre_trained_dict = {}
    for d in state_dicts:
        pre_trained_dict.update(d)
    return pre_trained_dict
