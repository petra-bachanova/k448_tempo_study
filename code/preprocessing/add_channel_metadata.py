import pandas as pd
import numpy as np
from pathlib import Path
from bids import BIDSLayout


def split_contacts(s: str) -> list[str]:
    """Split a string of semicolon-separated contacts into a list of string.

    Returns:
        list: list of contacts (str)
    """
    if pd.isna(s) or s == "":
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]


def process_channels(sub: str, ses: str, row: pd.Series, project_path: Path) -> None:
    """Process channels.tsv from a particular session from a participant.

    Args:
        sub (str): participant e.g. sub-17
        ses (str): session e.g. ses-01
        row (pd.Series): _description_
        project_path (Path): _description_
    """
    # set paths
    sub_ses_path = project_path / "rawdata" / sub / ses / "ieeg"
    channels_path = sub_ses_path / f"{sub}_{ses}_task-k448tempo_channels.tsv"
    # load channels.tsv from this sub-xx_ses-xx
    channels = pd.read_csv(channels_path, sep="\t", dtype={"name": "string"})

    # mark bad contacts and set status description
    bad_contacts = split_contacts(row.get("bad_contacts", np.nan))
    if bad_contacts:
        channels.loc[channels["name"].isin(bad_contacts), "status"] = "bad"
    channels["status_description"] = "manual: noisy or artifactual"

    # mark SOZ contacts
    include_soz = row.get("include_soz_analysis", 0)
    soz_contacts = split_contacts(row.get("soz_contacts", np.nan))
    if include_soz == 1 and soz_contacts:
        channels["soz_contact"] = np.where(
            channels["name"].isin(soz_contacts), "yes", "no"
        )
    else:
        channels["soz_contact"] = "no"

    # load MNE coordinates and anatomical location
    loc_path = project_path / "Electrode_locations" / f"{sub}_electrode_mni_and_annotation.xlsx"
    if loc_path.exists():
        locations = pd.read_excel(loc_path)
        locations = locations.rename(
            columns={
                "Contact": "name",
                "MNI1": "MNI_x",
                "MNI2": "MNI_y",
                "MNI3": "MNI_z",
                "Contact location": "contact_anatomy",
            }
        )

        # clean up contact names
        locations["name"] = (
            locations["name"].astype(str)
            # .str.replace("'", "", regex=False)
            .str.replace(r"'([A-Za-z]+)0*(\d+)'", r"\1\2", regex=True)
        )

        # merge locations into channels
        channels = channels.merge(locations, on="name", how="left")

    print("hi")
    # save back into participant’s folder
    channels.to_csv(f"{channels_path}", sep="\t", index=False)
    # channels.to_csv(f"{project_path}/test{sub}{ses}.tsv", sep="\t", index=False)


project_path = Path("/dartfs-hpc/rc/lab/E/ECoG/music_study_tempo")
# initialize PyBIDS layout & load participant, session metadata
layout = BIDSLayout(project_path / "rawdata", validate=False)
# TODO: rewrite paths to participants and session using layout
participants = pd.read_csv(project_path / "rawdata" / "participants.tsv", sep="\t")
sessions = pd.read_csv(project_path / "rawdata" / "sessions.tsv", sep="\t")
sessions = sessions[sessions["exclude"] == 0]

# merge metadata
meta = sessions.merge(participants, on="participant_id", how="left")

# iterate through (sub, ses) combinations
for _, row in meta.iterrows():
    sub, ses = row["participant_id"], row["session"]

    if sub not in ["sub-19", "sub-20"]:  # TODO need to check data from these subjects
        print(row)
        # check if channels.tsv exists
        kws = {
            "subject": sub.removeprefix("sub-"), 
            "session": ses.removeprefix("ses-"), 
            "suffix": "channels"}
        files = layout.get(**kws)
        if files:
            process_channels(sub, ses, row, project_path)
        else:
            print(f"No channels.tsv found for {sub}, {ses}")


# test_c = channels[channels["type"] == "SEEG"]
# in_channels = set(test_c["name"]).difference(locations["name"])
# in_channels = set([re.sub(r"\d+", "", i) for i in in_channels])

# in_locations = set(locations["name"]).difference(channels["name"])
# in_locations = set([re.sub(r"\d+", "", i) for i in in_locations])