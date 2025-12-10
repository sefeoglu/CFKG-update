```yaml
# Full Relation Taxonomy for FewRel Dataset
taxonomy:
  social:
    - follows
    - followed_by

  family:
    parent_child:
      parent_to_child: [father, mother, parent, family]
      child_to_parent: [child, son, daughter]
      descendant_relations: [son → child, daughter → child]
    sibling: [sibling]
    spouse: [spouse, husband, wife]

  location:
    administrative: 
      - located_in_the_administrative_territorial_entity
      - contains_administrative_territorial_entity
    general: [
      location, work_location, headquarters_location, residence,
      location_of_formation, located_on_terrain_feature,
      located_in_or_next_to_body_of_water, mountain_range,
      mouth_of_the_watercourse
    ]

  membership:
    membership: [member_of, member_of_political_party, league]
    participation: [participant, participant_of]

  ownership:
    - owned_by
    - manufacturer
    - developer
    - publisher
    - distributor
    - architect

  creative_work:
    contribution: [composer, performer, screenwriter]
    attributes: [characters, genre, notable_work]
    production: [record_label, original_network, publisher, distributor]
    crosswork_relations: [after_a_work_by, said_to_be_the_same_as]

  sports:
    activity: [sport, position_played_on_team_/_speciality]
    competition_structure: [competition_class, sports_season_of_league_or_competition, league]
    outcomes: [participating_team, winner, successful_candidate]

  jurisdiction:
    - applies_to_jurisdiction
    - licensed_to_broadcast_to
````